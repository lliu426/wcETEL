import numpy as np
from scipy.stats import beta
from scipy.spatial import Voronoi
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from shapely.geometry import Polygon, box, Point
from collections import defaultdict
from shapely import contains_xy
from scipy.interpolate import griddata



def finite_polygons_2d(vor, radius=1e6, bbox=None):
    if bbox is None:
        bbox = [-0.1, -0.1, 1.1, 1.1]
    new_regions = []
    new_vertices = vor.vertices.tolist()
    all_ridges = defaultdict(list)

    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges[p1].append((p2, v1, v2))
        all_ridges[p2].append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 not in region:
            polygon = [vor.vertices[i] for i in region]
            new_regions.append(Polygon(polygon).buffer(0))
            continue

        ridges = all_ridges[p1]
        new_region = []
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                new_region.append(v2)
                continue
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            far_point = vor.vertices[v2] + normal * radius
            new_vertices.append(far_point.tolist())
            new_region.append(v2)
            new_region.append(len(new_vertices) - 1)

        vs = np.array([new_vertices[v] for v in new_region])
        poly = Polygon(vs).buffer(0)
        poly = poly.intersection(box(*bbox))
        new_regions.append(poly)
    return new_regions



class PowerDiagram2D:
    def __init__(self, X, weights=None, L=1.0):
        self.X = np.array(X, copy=True)
        self.n = len(self.X)
        self.weights = np.zeros(self.n) if weights is None else np.array(weights, copy=True)
        self.L = L
        self.Cells = [None] * self.n
        self.indices = []
        self.updated_flag = False

    def update_laguerre_cells(self):
        vor = Voronoi(self.X)
        pad = 0.05 * self.L
        bbox = [-pad, -pad, self.L + pad, self.L + pad]
        polygons = finite_polygons_2d(vor, radius=1e3, bbox=bbox)
        self.Cells = [None] * self.n
        self.indices = []
        for i, poly in enumerate(polygons):
            if not poly.is_empty and poly.is_valid:
                self.Cells[i] = poly
                self.indices.append(i)
        self.updated_flag = True

    def compute_integrals(self, fun, grid_res=10):
        if not self.updated_flag:
            self.update_laguerre_cells()
    
        integrals = np.zeros(self.n)
        dx = self.L / grid_res
        area_per_cell = dx * dx
    
        xs = np.linspace(0, self.L, grid_res)
        ys = np.linspace(0, self.L, grid_res)
        Xg, Yg = np.meshgrid(xs, ys)
        grid_points = np.column_stack([Xg.ravel(), Yg.ravel()])  # (M^2, 2)
    
        domain = box(0, 0, self.L, self.L)
    
        for i in self.indices:
            cell = self.Cells[i].intersection(domain)
            if not cell.is_valid or cell.is_empty:
                continue
    
            # FAST vectorized check
            mask = contains_xy(cell, grid_points[:, 0], grid_points[:, 1])
            selected_pts = grid_points[mask]
            if selected_pts.size == 0:
                continue
    
            values = fun(selected_pts[:, 0], selected_pts[:, 1])
            integrals[i] = np.sum(values) * area_per_cell
    
        total = np.sum(integrals)
        if total > 0:
            integrals /= total  # optional normalization
        return integrals

    # def compute_integrals(self, fun):
    #     """
    #     Compute integrals over Laguerre cells assuming uniform density (Beta(1,1) × Beta(1,1)).
    #     This simply uses cell area as mass.
    #     """
    #     if not self.updated_flag:
    #         self.update_laguerre_cells()
    
    #     integrals = np.zeros(self.n)
    #     domain = box(0, 0, self.L, self.L)
    
    #     for i in self.indices:
    #         cell = self.Cells[i].intersection(domain)
    #         if not cell.is_valid or cell.is_empty:
    #             continue
    #         integrals[i] = cell.area  # uniform density -> mass = area (volume)
    
    #     total = np.sum(integrals)
    #     if total > 0:
    #         integrals /= total  # normalize to make it a probability vector
    #     return integrals



class OptimalTransport2D(PowerDiagram2D):
    def __init__(self, X, masses, rho, L=1.0):
        super().__init__(X, L=L)
        self.masses = np.array(masses, copy=True)
        self.rho = rho

    def compute_ot_cost(self, grid_res=10):
        if not self.updated_flag:
            self.update_laguerre_cells()
        integrals = np.zeros(self.n)
        dx = self.L / grid_res
        area_per_cell = dx * dx
        xs = np.linspace(0, self.L, grid_res)
        ys = np.linspace(0, self.L, grid_res)
        grid_points = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
        domain = box(0, 0, self.L, self.L)
        for i in self.indices:
            cell = self.Cells[i].intersection(domain)
            if not cell.is_valid or cell.is_empty:
                continue
            xi = self.X[i]
            mask = np.array([cell.contains(Point(pt)) for pt in grid_points])
            selected_pts = grid_points[mask]
            if selected_pts.size == 0:
                continue
            sq_dists = np.sum((selected_pts - xi)**2, axis=1)
            rho_vals = np.array([self.rho(x, y) for x, y in selected_pts])
            integrals[i] = np.sum((sq_dists - self.weights[i]) * rho_vals) * area_per_cell
        return np.sum(integrals) + np.sum(self.masses * self.weights)

    def update_weights(self, tol=1e-6, maxIter=50, verbose=False):
        alpha = 0.01
        tau_init = 0.5
        self.update_laguerre_cells()
        for it in range(maxIter):
            integrals = self.compute_integrals(self.rho)
            F = integrals - self.masses
            error = np.linalg.norm(F)
            if verbose:
                print(f"[{it}] Error: {error:.2e}")
            if error < tol:
                if verbose:
                    print("Converged!")
                break
            H = diags(np.ones(len(self.indices))).tocsc()
            try:
                dw_local = -spsolve(H, F[self.indices])
            except:
                dw_local = -F[self.indices]
            delta_w = np.zeros_like(self.weights)
            delta_w[self.indices] = dw_local * 0.5
            tau = tau_init
            cost_old = self.compute_ot_cost()
            for _ in range(10):
                new_weights = self.weights + tau * delta_w
                self.weights = new_weights
                self.update_laguerre_cells()
                try:
                    cost_new = self.compute_ot_cost()
                except:
                    cost_new = np.inf
                if np.isfinite(cost_new) and cost_new < cost_old:
                    break
                tau *= 0.5
            else:
                if verbose:
                    print("Line search failed.")
                break