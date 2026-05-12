import sys
import os
import numpy as np
from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import (
    Constant, Function, dirichletbc, form, locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    apply_lifting, assemble_matrix, assemble_vector,
    create_vector, create_matrix, set_bc,
)
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from ufl import (
    Identity, Measure, TestFunction, TrialFunction,
    as_vector, div, dot, grad, nabla_grad, inner, dx,
    lhs, rhs, FacetNormal,
)
from dolfinx.fem import assemble_scalar
from src.cfd.CFDParameters import CFDParameters


class CFDEnvSolver(CFDParameters):
    """
    Here we add the RL stepping logic to ensure we can train agent to do AFC on airfoil.

    Extra methods:
      boundary_conditions_env()  — same as parent but keeps jet BCs separate
      initialize()               — assembles matrices and solvers once 
      load_checkpoint(path)      — restores u_n, p_n from .npy each episode reset
      set_jet_velocities(Q)      — updates jet Dirichlet BCs with parabolic profiles
      step_n(n)                  — advances n IPCS timesteps, returns (C_D_arr, C_L_arr)
      sample_probes(probes)      — evaluates (u_x, u_y, p) at probe coords
    """

    def boundary_conditions_env(self, Re, U_m, oscillating=False):

        """
        Same as CFDParameters.boundary_conditions(), but jet strips get their own separate function-backed BCs so set_jet_velocities() can update them.
        """

        super().boundary_conditions(Re, U_m, oscillating)

        fdim = self.mesh.topology.dim - 1
        u_zero = np.zeros(self.mesh.geometry.dim, dtype=PETSc.ScalarType)

        self._u_jet_funcs = [Function(self.V) for _ in self.jet_markers]
        self.bcu_jets = []
        for u_jet, marker in zip(self._u_jet_funcs, self.jet_markers):
            facets = self.ft.find(marker)
            dofs   = locate_dofs_topological(self.V, fdim, facets)
            self.bcu_jets.append(dirichletbc(u_jet, dofs))

        airfoil_facets   = self.ft.find(self.airfoil_marker)
        bcu_airfoil_body = dirichletbc(
            u_zero,
            locate_dofs_topological(self.V, fdim, airfoil_facets),
            self.V
        )
        self.bcu = [self.bcu[0], self.bcu[1], bcu_airfoil_body] + self.bcu_jets

    def initialize(self):

        """
        Call once after boundary_conditions_env(). Survives across episode resets.
        """

        u  = TrialFunction(self.V);  v = TestFunction(self.V)
        p  = TrialFunction(self.Q);  q = TestFunction(self.Q)

        self._u_  = Function(self.V, name="Velocity")
        self._u_s = Function(self.V)
        self._u_n = Function(self.V)
        self._p_  = Function(self.Q, name="Pressure")
        self._p_n = Function(self.Q)

        F1  = (self.rho / self.k) * dot(u - self._u_n, v) * dx
        F1 += self.rho * dot(dot(self._u_n, nabla_grad(u)), v) * dx
        F1 += self.mu  * inner(nabla_grad(u), nabla_grad(v)) * dx
        F1 -= dot(self._p_n, div(v)) * dx
        self._a1 = form(lhs(F1))
        self._L1 = form(rhs(F1))

        self._a2 = form(dot(grad(p), grad(q)) * dx)
        self._L2 = form(dot(grad(self._p_n), grad(q)) * dx
                        - (self.rho / self.k) * div(self._u_s) * q * dx)

        self._a3 = form(self.rho * dot(u, v) * dx)
        self._L3 = form(self.rho * dot(self._u_s, v) * dx
                        - self.k * dot(grad(self._p_ - self._p_n), v) * dx)

        self._A1 = create_matrix(self._a1)
        self._b1 = create_vector(self.V)
        self._A2 = assemble_matrix(self._a2, bcs=self.bcp)
        self._A2.assemble()
        self._b2 = create_vector(self.Q)
        self._A3 = assemble_matrix(self._a3)
        self._A3.assemble()
        self._b3 = create_vector(self.V)

        self._solver1 = PETSc.KSP().create(self.mesh.comm)
        self._solver1.setType(PETSc.KSP.Type.BCGS)
        self._solver1.getPC().setType(PETSc.PC.Type.JACOBI)

        self._solver2 = PETSc.KSP().create(self.mesh.comm)
        self._solver2.setOperators(self._A2)
        self._solver2.setType(PETSc.KSP.Type.MINRES)
        pc2 = self._solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")

        self._solver3 = PETSc.KSP().create(self.mesh.comm)
        self._solver3.setOperators(self._A3)
        self._solver3.setType(PETSc.KSP.Type.CG)
        self._solver3.getPC().setType(PETSc.PC.Type.SOR)

        # L/D
        n_hat         = -FacetNormal(self.mesh)
        ds_airfoil    = Measure("ds", domain=self.mesh, subdomain_data=self.ft,
                                subdomain_id=self.airfoil_marker)
        sigma         = (-self._p_ * Identity(self.mesh.geometry.dim)
                         + self.mu * (grad(self._u_) + grad(self._u_).T))
        self._F_D_form = form(dot(sigma * n_hat, as_vector([1.0, 0.0])) * ds_airfoil)
        self._F_L_form = form(dot(sigma * n_hat, as_vector([0.0, 1.0])) * ds_airfoil)
        self._denom    = 0.5 * float(self.rho.value) * self.U_bar ** 2

        self._bb_tree = bb_tree(self.mesh, self.mesh.topology.dim)

        print("CFDEnvSolver initialized — matrices and solvers ready.")

    def load_checkpoint(self, path):
        """
        Restore u_n, p_n from .npy checkpoint.
        """
        p = Path(path)
        u_arr = np.load(p / "u.npy")
        p_arr = np.load(p / "p.npy")

        self._u_n.x.array[:] = u_arr;  self._u_n.x.scatter_forward()
        self._p_n.x.array[:] = p_arr;  self._p_n.x.scatter_forward()
        self._u_.x.array[:]  = u_arr;  self._u_.x.scatter_forward()
        self._p_.x.array[:]  = p_arr;  self._p_.x.scatter_forward()

        self.t = float(np.load(p / "t_array.npy")[-1])

    def set_jet_velocities(self, Q: np.ndarray, jet_normals: list):

        """
        Update jet Dirichlet BCs with parabolic profiles scaled to volumetric flow Q_j.
        """

        fdim = self.mesh.topology.dim - 1

        for u_jet, marker, normal, q_k in zip(
                self._u_jet_funcs, self.jet_markers, jet_normals, Q):

            facets     = self.ft.find(marker)
            jet_dofs   = locate_dofs_topological(self.V, fdim, facets)
            dof_coords = self.V.tabulate_dof_coordinates()[jet_dofs]
            tangent    = np.array([-normal[1], normal[0]])
            s          = dof_coords[:, :2] @ tangent
            s_min, s_max = float(s.min()), float(s.max())
            jet_width  = max(s_max - s_min, 1e-12)

            _s_min, _s_max, _w, _q, _n = s_min, s_max, jet_width, float(q_k), np.array(normal)

            def profile(x, s_min=_s_min, s_max=_s_max, w=_w, q=_q, n=_n, t=tangent):
                s      = x[0] * t[0] + x[1] * t[1]
                s_norm = np.clip((s - s_min) / (s_max - s_min + 1e-12), 0.0, 1.0)
                weight = 6.0 * s_norm * (1.0 - s_norm)   # integrates to 1 over [0,1]
                u_peak = q / w
                vals   = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
                vals[0] = weight * u_peak * n[0]
                vals[1] = weight * u_peak * n[1]
                return vals

            u_jet.interpolate(profile)
            u_jet.x.scatter_forward()

    def step_n(self, n: int):
        """
        Advance n IPCS timesteps with current BCs.
        Returns arrays C_D (n,) and C_L (n,).
        """
        C_D = np.zeros(n)
        C_L = np.zeros(n)

        for i in range(n):
            self.t += self.dt
            self.inlet_velocity.t = self.t
            self.u_inlet.interpolate(self.inlet_velocity)
            self.u_inlet.x.scatter_forward()

            # Step 1
            self._A1.zeroEntries()
            assemble_matrix(self._A1, self._a1, bcs=self.bcu)
            self._A1.assemble()
            self._solver1.setOperators(self._A1)
            with self._b1.localForm() as loc: loc.set(0)
            assemble_vector(self._b1, self._L1)
            apply_lifting(self._b1, [self._a1], [self.bcu])
            self._b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                                  mode=PETSc.ScatterMode.REVERSE)
            set_bc(self._b1, self.bcu)
            self._solver1.solve(self._b1, self._u_s.x.petsc_vec)
            self._u_s.x.scatter_forward()

            # Step 2
            with self._b2.localForm() as loc: loc.set(0)
            assemble_vector(self._b2, self._L2)
            apply_lifting(self._b2, [self._a2], [self.bcp])
            self._b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                                  mode=PETSc.ScatterMode.REVERSE)
            set_bc(self._b2, self.bcp)
            self._solver2.solve(self._b2, self._p_.x.petsc_vec)
            self._p_.x.scatter_forward()

            # Step 3
            with self._b3.localForm() as loc: loc.set(0)
            assemble_vector(self._b3, self._L3)
            self._b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                                  mode=PETSc.ScatterMode.REVERSE)
            self._solver3.solve(self._b3, self._u_.x.petsc_vec)
            self._u_.x.scatter_forward()

            F_D = self.mesh.comm.allreduce(assemble_scalar(self._F_D_form), op=MPI.SUM)
            F_L = self.mesh.comm.allreduce(assemble_scalar(self._F_L_form), op=MPI.SUM)
            C_D[i] = F_D / self._denom
            C_L[i] = F_L / self._denom

            self._u_n.x.array[:] = self._u_.x.array[:]
            self._u_n.x.scatter_forward()
            self._p_n.x.array[:] = self._p_.x.array[:]
            self._p_n.x.scatter_forward()

        return C_D, C_L

    def sample_probes(self, probes: np.ndarray, grid_shape: tuple):
        """
        Interpolate u_x, u_y, p at probe locations.

        probes     (N, 2) array of (x, y) coordinates.
        grid_shape (H, W) — e.g. (12, 16) for fine probes.

        Returns obs of shape (3, H, W): channels [u_x, u_y, p].
        
        """
        pts = np.column_stack([probes, np.zeros(len(probes))])  # (N, 3)

        candidates     = compute_collisions_points(self._bb_tree, pts)
        colliding      = compute_colliding_cells(self.mesh, candidates, pts)
        cell_per_point = np.array([
            colliding.links(i)[0] if len(colliding.links(i)) > 0 else 0
            for i in range(len(pts))
        ], dtype=np.int32)

        u_vals = self._u_.eval(pts, cell_per_point)   # (N, 2)
        p_vals = self._p_.eval(pts, cell_per_point)   # (N, 1)

        H, W = grid_shape
        obs = np.stack([
            u_vals[:, 0].reshape(H, W),
            u_vals[:, 1].reshape(H, W),
            p_vals[:, 0].reshape(H, W),
        ], axis=0).astype(np.float32)           # (3, H, W)

        return obs
