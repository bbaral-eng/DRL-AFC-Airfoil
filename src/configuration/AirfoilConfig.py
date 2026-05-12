import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import subprocess

JET_WIDTH   = 0.01                                      # width of each jet strip (in chord units)
JET_MIN_SEP = 0.01                                      # minimum separation between jet centers (in chord units) 
JET_COLORS  = ["#004300", "#28BE28", "#00FF00"]   # for plotting jets


class AirfoilConfig:
    def __init__(self, alpha, NACA_file, jet_locs, probe_density="fine"):

        """
        alpha:         angle of attack in degrees
        NACA_file:     path to .dat file 
        jet_locs:      list of 3 chord-wise x-positions for jets on the upper surface (between 0 and 1)
        probe_density: "fine" (16x12 = 192 probes) or "coarse" (8x6 = 48 probes)
        """

        if not 0 <= alpha <= 90:
            raise ValueError("Angle of attack must be between 0 and 90 degrees")
        if probe_density not in ("fine", "coarse"):
            raise ValueError("probe_density must be 'fine' or 'coarse'")
        if len(jet_locs) != 3:
            raise ValueError("jet_locs must contain 3 positions")

        jet_locs = sorted(float(j) for j in jet_locs)
        for j in jet_locs:
            if not (JET_WIDTH / 2 < j < 1.0 - JET_WIDTH / 2):
                raise ValueError(f"Jet at {j} is too close to the leading or trailing edge")
        for a, b in zip(jet_locs, jet_locs[1:]):
            if b - a < JET_MIN_SEP:
                raise ValueError(f"Jets at {a} and {b} are less than {JET_MIN_SEP} apart")

        self.alpha = alpha
        self.NACA_file = Path(NACA_file)
        self.jet_locs = jet_locs
        self.probe_density = probe_density
        self.geo_filename = None
        self.msh_filename = None

        self.load_airfoil()
        self._insert_jet_breakpoints()   
        self.rotate_airfoil()
        self._compute_jet_normals()
        self._compute_probes()
        self.plot_airfoil()
        self.generate_mesh()

    def load_airfoil(self):
        data = np.loadtxt(self.NACA_file, skiprows=1)
        self.x_coords = data[:, 0]
        self.y_coords = data[:, 1]

    def _insert_jet_breakpoints(self):
        
        upper = self.y_coords >= 0
        ux, uy = self.x_coords[upper], self.y_coords[upper]

        aug_x, aug_y = [ux[0]], [uy[0]]
        for i in range(len(ux) - 1):
            x_lo, x_hi = min(ux[i], ux[i + 1]), max(ux[i], ux[i + 1])
            for jc in self.jet_locs:
                for edge in (jc - JET_WIDTH / 2, jc + JET_WIDTH / 2):
                    if x_lo < edge < x_hi:
                        t = (edge - ux[i]) / (ux[i + 1] - ux[i])
                        aug_x.append(ux[i] + t * (ux[i + 1] - ux[i]))
                        aug_y.append(uy[i] + t * (uy[i + 1] - uy[i]))
            aug_x.append(ux[i + 1])
            aug_y.append(uy[i + 1])

        order = np.argsort(-np.array(aug_x))
        aug_x = np.array(aug_x)[order]
        aug_y = np.array(aug_y)[order]

        lx = self.x_coords[~upper]
        ly = self.y_coords[~upper]
        self.x_coords = np.concatenate([aug_x, lx])
        self.y_coords = np.concatenate([aug_y, ly])

        half = JET_WIDTH / 2
        self.jet_segment_indices = []
        for jc in self.jet_locs:
            in_jet = np.where((aug_x >= jc - half) & (aug_x <= jc + half))[0]
            self.jet_segment_indices.append((int(in_jet[0]), int(in_jet[-1])))

    def rotate_airfoil(self):
        x, y = self.x_coords.copy(), self.y_coords.copy()
        a = np.radians(-self.alpha)
        self.x_coords = x * np.cos(a) - y * np.sin(a)
        self.y_coords = x * np.sin(a) + y * np.cos(a)

    def _compute_jet_normals(self):

        """Outward unit normal at the midpoint of each jet (in the rotated frame)."""

        self.jet_normals = []
        for start, end in self.jet_segment_indices:
            mid = (start + end) // 2
            prev = np.array([self.x_coords[mid - 1], self.y_coords[mid - 1]])
            nxt  = np.array([self.x_coords[mid + 1], self.y_coords[mid + 1]])
            tangent = (nxt - prev) / np.linalg.norm(nxt - prev)
            normal  = np.array([-tangent[1], tangent[0]])
            if normal[1] < 0:
                normal = -normal   
            self.jet_normals.append(normal)

    def _compute_probes(self):
        if self.probe_density == "fine":
            x, y = np.meshgrid(np.linspace(1.0, 2.5, 16), np.linspace(-0.5, 0.5, 12))
        else:
            x, y = np.meshgrid(np.linspace(1.0, 2.5, 8), np.linspace(-0.5, 0.5, 6))
        self.probes = np.column_stack([x.ravel(), y.ravel()])

    def plot_airfoil(self):
        plt.figure(figsize=(10, 5))
        plt.gca().set_axisbelow(True)
        plt.grid(True)
        plt.fill(self.x_coords, self.y_coords, color='white', edgecolor='blue', linewidth=1)

        for k, (start, end) in enumerate(self.jet_segment_indices):
            plt.plot(self.x_coords[start:end + 1], self.y_coords[start:end + 1],
                     color=JET_COLORS[k], linewidth=4, label=f'jet {k + 1}')

        plt.scatter(self.probes[:, 0], self.probes[:, 1],
                    s=8, c='red', alpha=0.7, label=f'{self.probe_density} probes ({len(self.probes)})')
        plt.legend(fontsize=8)
        plt.title(f'Airfoil at AoA = {self.alpha}°')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(-0.5, 3)
        plt.ylim(-0.7, 0.7)
        plt.show()

    def _jet_tag(self, i):
        """Return jet index (0/1/2) if point i is inside a jet strip, else None."""
        for k, (s, e) in enumerate(self.jet_segment_indices):
            if s <= i <= e:
                return k
        return None

    def generate_mesh(self):
        out_dir = Path(__file__).parent / "geo_and_mesh"
        out_dir.mkdir(exist_ok=True)
        self.geo_filename = str(out_dir / f"{self.NACA_file.stem}_{self.alpha}.geo")
        self.msh_filename = str(out_dir / f"{self.NACA_file.stem}_{self.alpha}.msh")

        n_pts = len(self.x_coords)

        segments = []
        current_tag = self._jet_tag(0)
        current_pts = [0]
        for i in range(1, n_pts):
            t = self._jet_tag(i)
            if t == current_tag:
                current_pts.append(i)
            else:
                segments.append((current_tag, current_pts))
                current_tag = t
                current_pts = [current_pts[-1], i]   # shared boundary point
        segments.append((current_tag, current_pts))

        with open(self.geo_filename, "w") as f:
            f.write('SetFactory("OpenCASCADE");\n\n')

            # bounding box
            f.write("Point(1) = {-0.5, -0.7, 0, 0.03};\n")
            f.write("Point(2) = {3,    -0.7, 0, 0.03};\n")
            f.write("Point(3) = {3,     0.7, 0, 0.03};\n")
            f.write("Point(4) = {-0.5,  0.7, 0, 0.03};\n")
            f.write("Line(1) = {1,2}; Line(2) = {2,3}; Line(3) = {3,4}; Line(4) = {4,1};\n")
            f.write("Curve Loop(10) = {1,2,3,4};\n\n")

            # airfoil points
            for i, (xi, yi) in enumerate(zip(self.x_coords, self.y_coords)):
                size = 0.001 if self._jet_tag(i) is not None else 0.005
                f.write(f"Point({1000 + i}) = {{{xi:.8f}, {yi:.8f}, 0, {size}}};\n")

            # one BSpline per contiguous segment
            curve_id  = 2000
            body_curves = []
            jet_curves  = {0: [], 1: [], 2: []}

            for tag, pts in segments:
                if len(pts) < 2:
                    continue
                pt_str = ", ".join(str(1000 + p) for p in pts)
                f.write(f"BSpline({curve_id}) = {{{pt_str}}};\n")
                if tag is None:
                    body_curves.append(curve_id)
                else:
                    jet_curves[tag].append(curve_id)
                curve_id += 1

            # closing line at trailing edge if the profile is open
            all_curves = list(range(2000, curve_id))
            first = np.array([self.x_coords[0],          self.y_coords[0]])
            last  = np.array([self.x_coords[n_pts - 1],  self.y_coords[n_pts - 1]])
            if not np.allclose(first, last, atol=1e-6):
                f.write(f"Line({curve_id}) = {{{1000 + n_pts - 1}, 1000}};\n")
                body_curves.append(curve_id)
                all_curves.append(curve_id)
                curve_id += 1

            loop_str = ", ".join(str(c) for c in all_curves)
            f.write(f"\nCurve Loop(20) = {{{loop_str}}};\n")
            f.write("Plane Surface(30) = {10, 20};\n\n")

            # physical groups
            f.write('Physical Curve("inlet")          = {4};\n')
            f.write('Physical Curve("outlet")         = {2};\n')
            f.write('Physical Curve("top_and_bottom") = {1, 3};\n')
            f.write(f'Physical Curve("airfoil") = {{{", ".join(str(c) for c in body_curves)}}};\n')
            for k in range(3):
                f.write(f'Physical Curve("jet_{k+1}") = {{{", ".join(str(c) for c in jet_curves[k])}}};\n')
            f.write('Physical Surface("fluid") = {30};\n\n')

            # mesh size field
            all_str = ", ".join(str(c) for c in all_curves)
            f.write(f"Field[1] = Distance;\nField[1].CurvesList = {{{all_str}}};\n")
            f.write("Field[1].NumPointsPerCurve = 100;\n")
            f.write("Field[2] = Threshold;\nField[2].InField = 1;\n")
            f.write("Field[2].SizeMin = 0.001;\nField[2].SizeMax = 0.025;\n")
            f.write("Field[2].DistMin = 0.0;\nField[2].DistMax = 0.4;\n")
            f.write("Background Field = 2;\n")
            f.write("Mesh.ElementOrder = 2;\nMesh.Optimize = 1;\nMesh.OptimizeNetgen = 1;\n")

        subprocess.run(["gmsh", "-2", self.geo_filename, "-o", self.msh_filename], check=True)
        print(f"Mesh generated: {self.msh_filename}")
        print(f"  Jet normals: {[list(np.round(n, 4)) for n in self.jet_normals]}")
