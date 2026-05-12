import gmsh
import numpy as np
import tqdm.autonotebook

from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element
from dolfinx.fem import (
    Constant, Function, functionspace,
    assemble_scalar, dirichletbc, form, locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    apply_lifting, assemble_matrix, assemble_vector,
    create_vector, create_matrix, set_bc,
)
from dolfinx.io import XDMFFile, gmsh as gmshio
from ufl import (
    FacetNormal, Identity, Measure, TestFunction, TrialFunction,
    as_vector, div, dot, dx, inner, lhs, grad, nabla_grad, rhs,
)
import matplotlib.pyplot as plt

class CFDParameters:

    def __init__(self, T, dt, mu, rho, mesh_filename):

        """
        T  (float): Final simulation time in seconds.
        dt (int):   Time-steps per second  (actual dt = 1/dt).
        mu (float): Dynamic viscosity.
        rho (float): Fluid density.
        mesh_filename (str): Path to the .msh file produced by AirfoilConfig.
        """

        self.t = 0.0
        self.T = T

        if not (isinstance(dt, int) and dt > 0):
            raise ValueError("dt must be a positive int (steps per second).")

        self.dt       = 1.0 / dt
        self.num_steps = int(T * dt)

        gmsh.initialize()
        gmsh.open(mesh_filename)

        boundary_markers = {}
        for dim, tag in gmsh.model.getPhysicalGroups(dim=1):
            boundary_markers[gmsh.model.getPhysicalName(dim, tag)] = tag

        mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2)
        gmsh.finalize()

        self.mesh = mesh_data.mesh
        self.ft   = mesh_data.facet_tags
        self.ft.name = "Facet markers"

        self.inlet_marker   = boundary_markers["inlet"]
        self.outlet_marker  = boundary_markers["outlet"]
        self.wall_marker    = boundary_markers["top_and_bottom"]
        self.airfoil_marker = boundary_markers["airfoil"]
        self.jet_markers    = [boundary_markers[f"jet_{k+1}"] for k in range(3)]

        self.k   = Constant(self.mesh, PETSc.ScalarType(self.dt))
        self.mu  = Constant(self.mesh, PETSc.ScalarType(mu))
        self.rho = Constant(self.mesh, PETSc.ScalarType(rho))

        print(f"Mesh loaded — {self.mesh.topology.index_map(0).size_global} vertices, "
              f"T={T}s, steps={self.num_steps}, dt={self.dt:.6f}")

    def boundary_conditions(self, Re, U_m, oscillating=False):
        
        """
        Set up Taylor-Hood (P2/P1) function spaces and boundary conditions.

        Re (float):         Reynolds number.
        U_m (float):        Peak inlet velocity.
        oscillating (bool): If True, modulates inlet as 1 + 0.25*sin(2πt).

        Boundary conditions:
          inlet        — parabolic profile, optionally oscillating, linearly ramped
          top/bottom   — no-slip
          airfoil body — no-slip  (jet strips also treated as no-slip here)
          outlet       — p = 0
        """

        v_cg2 = element("Lagrange", self.mesh.basix_cell(), 2, shape=(self.mesh.geometry.dim,))
        s_cg1 = element("Lagrange", self.mesh.basix_cell(), 1)
        self.V = functionspace(self.mesh, v_cg2)
        self.Q = functionspace(self.mesh, s_cg1)

        fdim = self.mesh.topology.dim - 1

        U_bar    = (2.0 / 3.0) * U_m
        nu_actual = float(self.mu.value) / float(self.rho.value)
        nu_target = U_bar / Re
        if not np.isclose(nu_target, nu_actual, rtol=1e-3):
            raise ValueError(
                f"Re={Re}, U_m={U_m} implies nu={nu_target:.6f}, but mu/rho={nu_actual:.6f}."
            )
        self.U_bar = U_bar

        D, H = 1.0, 1.4

        class InletVelocity:
            def __init__(self, t, U_m, oscillating):
                self.t, self.U_m, self.oscillating = t, U_m, oscillating

            def __call__(self, x):
                f_t = (1.0 + 0.25 * np.sin(2 * np.pi * self.t)) if self.oscillating else 1.0
                values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
                values[0] = (4 * self.U_m * (0.7 * D + x[1]) * (0.7 * D - x[1])) / H**2 * f_t
                return values

        self.inlet_velocity = InletVelocity(self.t, U_m, oscillating)
        self.u_inlet = Function(self.V)
        self.u_inlet.interpolate(self.inlet_velocity)

        bcu_inflow = dirichletbc(
            self.u_inlet,
            locate_dofs_topological(self.V, fdim, self.ft.find(self.inlet_marker))
        )

        u_zero = np.zeros(self.mesh.geometry.dim, dtype=PETSc.ScalarType)
        bcu_walls = dirichletbc(
            u_zero,
            locate_dofs_topological(self.V, fdim, self.ft.find(self.wall_marker)), self.V
        )

        # Airfoil body + jet strips → all no-slip for uncontrolled flow
        all_airfoil_facets = np.concatenate([
            self.ft.find(self.airfoil_marker),
            *[self.ft.find(m) for m in self.jet_markers]
        ])
        bcu_airfoil = dirichletbc(
            u_zero,
            locate_dofs_topological(self.V, fdim, all_airfoil_facets), self.V
        )

        self.bcu = [bcu_inflow, bcu_walls, bcu_airfoil]

        bcp_outlet = dirichletbc(
            PETSc.ScalarType(0),
            locate_dofs_topological(self.Q, fdim, self.ft.find(self.outlet_marker)), self.Q
        )
        self.bcp = [bcp_outlet]

        print(f"BCs set — U_m={U_m}, U_bar={U_bar:.4f}, Re={Re}, nu={nu_actual:.6f}")

    def IPCS(self, output_dir, save_interval=10, t_ramp=2.0, name="initial_flow"):

        """
        Run IPCS from t=0 to T.  Writes XDMF to output_dir every save_interval steps.
        Requires boundary_conditions() to have been called first.

        output_dir (str):   Directory for XDMF output (created if missing).
        save_interval (int): Write fields every N steps.
        t_ramp (float):     Seconds over which inlet ramps from 0 → U_m.
        name (str):         Prefix for output filenames.
        """
        
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        u   = TrialFunction(self.V);  v = TestFunction(self.V)
        p   = TrialFunction(self.Q);  q = TestFunction(self.Q)

        u_  = Function(self.V, name="Velocity")
        u_s = Function(self.V)
        u_n = Function(self.V)
        p_  = Function(self.Q, name="Pressure")
        p_n = Function(self.Q)

        # IPCS variational forms
        F1  = (self.rho / self.k) * dot(u - u_n, v) * dx
        F1 += self.rho * dot(dot(u_n, nabla_grad(u)), v) * dx
        F1 += self.mu  * inner(nabla_grad(u), nabla_grad(v)) * dx
        F1 -= dot(p_n, div(v)) * dx
        a1, L1 = form(lhs(F1)), form(rhs(F1))

        a2 = form(dot(grad(p), grad(q)) * dx)
        L2 = form(dot(grad(p_n), grad(q)) * dx - (self.rho / self.k) * div(u_s) * q * dx)

        a3 = form(self.rho * dot(u, v) * dx)
        L3 = form(self.rho * dot(u_s, v) * dx - self.k * dot(grad(p_ - p_n), v) * dx)

        A1 = create_matrix(a1);  b1 = create_vector(self.V)
        A2 = assemble_matrix(a2, bcs=self.bcp);  A2.assemble();  b2 = create_vector(self.Q)
        A3 = assemble_matrix(a3);  A3.assemble();  b3 = create_vector(self.V)

        solver1 = PETSc.KSP().create(self.mesh.comm)
        solver1.setType(PETSc.KSP.Type.BCGS)
        solver1.getPC().setType(PETSc.PC.Type.JACOBI)

        solver2 = PETSc.KSP().create(self.mesh.comm)
        solver2.setOperators(A2)
        solver2.setType(PETSc.KSP.Type.MINRES)
        pc2 = solver2.getPC(); pc2.setType(PETSc.PC.Type.HYPRE); pc2.setHYPREType("boomeramg")

        solver3 = PETSc.KSP().create(self.mesh.comm)
        solver3.setOperators(A3)
        solver3.setType(PETSc.KSP.Type.CG)
        solver3.getPC().setType(PETSc.PC.Type.SOR)

        # lift/drag
        n_hat      = -FacetNormal(self.mesh)
        ds_airfoil = Measure("ds", domain=self.mesh, subdomain_data=self.ft,
                             subdomain_id=self.airfoil_marker)
        sigma   = -p_ * Identity(self.mesh.geometry.dim) + self.mu * (grad(u_) + grad(u_).T)
        F_D_form = form(dot(sigma * n_hat, as_vector([1.0, 0.0])) * ds_airfoil)
        F_L_form = form(dot(sigma * n_hat, as_vector([0.0, 1.0])) * ds_airfoil)
        denom    = 0.5 * float(self.rho.value) * self.U_bar ** 2

        self.C_D     = np.zeros(self.num_steps)
        self.C_L     = np.zeros(self.num_steps)
        self.t_array = np.zeros(self.num_steps)

        # P2 pressure space for XDMF output
        p_out = Function(functionspace(self.mesh, element("Lagrange", self.mesh.basix_cell(), 2)),
                         name="Pressure")
        xdmf_u = XDMFFile(self.mesh.comm, f"{output_dir}/velocity_{name}.xdmf", "w")
        xdmf_p = XDMFFile(self.mesh.comm, f"{output_dir}/pressure_{name}.xdmf", "w")
        xdmf_u.write_mesh(self.mesh)
        xdmf_p.write_mesh(self.mesh)

        progress = tqdm.autonotebook.tqdm(desc="IPCS", total=self.num_steps)
        for i in range(self.num_steps):
            progress.update(1)
            self.t += self.dt

            ramp = min(self.t / t_ramp, 1.0) if t_ramp > 0.0 else 1.0
            self.inlet_velocity.t = self.t
            self.u_inlet.interpolate(self.inlet_velocity)
            self.u_inlet.x.array[:] *= ramp
            self.u_inlet.x.scatter_forward()

            # Step 1 — tentative velocity
            A1.zeroEntries()
            assemble_matrix(A1, a1, bcs=self.bcu)
            A1.assemble()
            solver1.setOperators(A1)
            with b1.localForm() as loc: loc.set(0)
            assemble_vector(b1, L1)
            apply_lifting(b1, [a1], [self.bcu])
            b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b1, self.bcu)
            solver1.solve(b1, u_s.x.petsc_vec)
            u_s.x.scatter_forward()

            # Step 2 — pressure correction
            with b2.localForm() as loc: loc.set(0)
            assemble_vector(b2, L2)
            apply_lifting(b2, [a2], [self.bcp])
            b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b2, self.bcp)
            solver2.solve(b2, p_.x.petsc_vec)
            p_.x.scatter_forward()

            # Step 3 — velocity correction
            with b3.localForm() as loc: loc.set(0)
            assemble_vector(b3, L3)
            b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            solver3.solve(b3, u_.x.petsc_vec)
            u_.x.scatter_forward()

            if i % save_interval == 0:
                p_out.interpolate(p_)
                xdmf_u.write_function(u_, self.t)
                xdmf_p.write_function(p_out, self.t)

            F_D = self.mesh.comm.allreduce(assemble_scalar(F_D_form), op=MPI.SUM)
            F_L = self.mesh.comm.allreduce(assemble_scalar(F_L_form), op=MPI.SUM)
            self.C_D[i]     = F_D / denom
            self.C_L[i]     = F_L / denom
            self.t_array[i] = self.t

            u_n.x.array[:] = u_.x.array[:]
            u_n.x.scatter_forward()
            p_n.x.array[:] = p_.x.array[:]
            p_n.x.scatter_forward()

        self.u_field = u_
        self.p_field = p_

        progress.close()
        xdmf_u.close()
        xdmf_p.close()
        for obj in [A1, A2, A3, b1, b2, b3, solver1, solver2, solver3]:
            obj.destroy()

        print(f"IPCS complete — results in '{output_dir}/'")

    def save_checkpoint(self, path):
        """
        Save the final velocity/pressure fields and C_D/C_L history to disk.
        These are used to warm-start each RL episode from a fully-developed flow state.

        path (str): Directory to write checkpoint files into.
        """
        p = Path(path)
        p.mkdir(exist_ok=True, parents=True)
        np.save(p / "u.npy",       self.u_field.x.array)
        np.save(p / "p.npy",       self.p_field.x.array)
        np.save(p / "C_D.npy",     self.C_D)
        np.save(p / "C_L.npy",     self.C_L)
        np.save(p / "t_array.npy", self.t_array)
        print(f"Checkpoint saved to '{p}/'")

    def plot_results(self):

        beta = self.C_L / np.where(np.abs(self.C_D) > 1e-12, self.C_D, 1.0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(self.t_array, self.C_D)
        axes[0].set(xlabel="t (s)", ylabel="$C_D$", title="Drag Coefficient")
        axes[0].grid(True)

        axes[1].plot(self.t_array, self.C_L)
        axes[1].set(xlabel="t (s)", ylabel="$C_L$", title="Lift Coefficient")
        axes[1].grid(True)

        axes[2].plot(self.t_array, beta)
        axes[2].set(xlabel="t (s)", ylabel=r"$\beta = C_L/C_D$", title="Lift-to-Drag Ratio")
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()
