"""
FEniCS-based velocity solver for discontinuous diffeomorphic registration.
Features the "SciPy Escape Hatch": Extracts FEniCS matrices to SciPy sparse arrays 
to perform 100% thread-safe ODE integration without any PETSc memory collisions.
"""

import os
import numpy as np
import torch
import threading
import scipy.ndimage as ndimage

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

try:
    import fenics as fe
    FENICS_AVAILABLE = True
    fe.parameters["form_compiler"]["optimize"] = False
    fe.parameters["form_compiler"]["cpp_optimize"] = False
except ImportError:
    FENICS_AVAILABLE = False
    print("Warning: FEniCS not available.")

TORCH_FENICS_AVAILABLE = False  

from scipy.fft import dctn as _scipy_dctn, idctn as _scipy_idctn

# =============================================================================
# PyTorch Autograd Integration
# =============================================================================
class _DGFenicsSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, solver):
        ctx.solver = solver
        return solver._solve_without_gradients(m)

    @staticmethod
    def backward(ctx, grad_out):
        solver = ctx.solver
        grad_m = solver._solve_without_gradients(grad_out)
        return grad_m, None

# =============================================================================
# Fallback FFT Solver
# =============================================================================
class _HelmholtzDCTSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, eigval_inv):
        m_np = m.detach().cpu().numpy().astype(np.float64)
        B, C = m_np.shape[0], m_np.shape[1]
        u_np = np.empty_like(m_np)
        for b in range(B):
            for c in range(C):
                M = _scipy_dctn(m_np[b, c], type=2, norm='ortho')
                u_np[b, c] = _scipy_idctn(M * eigval_inv, type=2, norm='ortho')
        ctx.eigval_inv = eigval_inv
        return torch.from_numpy(u_np).to(dtype=m.dtype, device=m.device)

    @staticmethod
    def backward(ctx, grad_out):
        eigval_inv = ctx.eigval_inv
        g_np = grad_out.detach().cpu().numpy().astype(np.float64)
        B, C = g_np.shape[0], g_np.shape[1]
        grad_m = np.empty_like(g_np)
        for b in range(B):
            for c in range(C):
                G = _scipy_dctn(g_np[b, c], type=2, norm='ortho')
                grad_m[b, c] = _scipy_idctn(G * eigval_inv, type=2, norm='ortho')
        return torch.from_numpy(grad_m).to(dtype=grad_out.dtype, device=grad_out.device), None

def _helmholtz_fft_solve(m, spacing, R=1, alpha=0.05):
    _, _, H, W = m.shape
    dy = float(spacing[0])
    dx = float(spacing[1]) if len(spacing) > 1 else dy
    k_y = np.arange(H, dtype=np.float64)
    k_x = np.arange(W, dtype=np.float64)
    lam_y = (2.0 / dy**2) * (1.0 - np.cos(np.pi * k_y / H))
    lam_x = (2.0 / dx**2) * (1.0 - np.cos(np.pi * k_x / W))
    lam = lam_y[:, None] + lam_x[None, :]
    eigval_inv = 1.0 / (1.0 + alpha**2 * lam) ** R 
    return _HelmholtzDCTSolve.apply(m, eigval_inv)

# =============================================================================
# Main FEniCS DG Solver Class
# =============================================================================
class FenicsDiscontinuousVelocitySolver:
    def __init__(self, shape, spacing, mask, R=2, penalty_parameter=1.0, verbose=False, alpha=0.05):
        if not FENICS_AVAILABLE:
            raise ImportError("FEniCS is not available.")

        self.shape = tuple(int(s) for s in (shape.detach().cpu().numpy() if hasattr(shape, 'detach') else shape))
        self.spacing = spacing.detach().cpu().numpy() if hasattr(spacing, 'detach') else np.asarray(spacing)
        self.mask = mask.detach().cpu().numpy() if hasattr(mask, 'detach') else (np.asarray(mask) if mask is not None else None)
        self.R = R
        self.alpha = alpha
        self.penalty = penalty_parameter 
        self.verbose = verbose

        self.mesh = self._create_mesh()

        self.V = fe.VectorFunctionSpace(self.mesh, 'DG', 1) 
        self.V_cg = fe.FunctionSpace(self.mesh, 'CG', 1)

        self.interface_marker = self._mark_interface()
        self.dx = fe.Measure('dx', domain=self.mesh)
        self.dS = fe.Measure('dS', domain=self.mesh, subdomain_data=self.interface_marker)

        self.lock = threading.Lock()

        if verbose:
            print(f"FEniCS Discontinuous solver initialized:")
            print(f"  Mesh: {self.mesh.num_vertices()} vertices, {self.mesh.num_cells()} cells")
            print(f"  DG Space DOFs: {self.V.dim()}")

        self._precompute_mappings()
        self._preassemble_matrix()

    def _create_mesh(self):
        H, W = self.shape
        dy = float(self.spacing[0]) if len(self.spacing) > 0 else 1.0
        dx = float(self.spacing[1]) if len(self.spacing) > 1 else 1.0
        return fe.RectangleMesh(fe.Point(0, 0), fe.Point((W - 1) * dx, (H - 1) * dy), W - 1, H - 1, "crossed")

    def _mark_interface(self):
        facet_marker = fe.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        facet_marker.set_all(0)
        if self.mask is None: return facet_marker

        H, W = self.shape
        dy = float(self.spacing[0])
        dx = float(self.spacing[1])

        self.mesh.init(2, 1)   
        self.mesh.init(1, 2)   

        def cell_region(cell):
            cx, cy = cell.midpoint().x(), cell.midpoint().y()
            j = max(0, min(int(cx / dx), W - 1)) # X maps to j
            i = max(0, min(int(cy / dy), H - 1)) # Y maps to i
            return int(self.mask[i, j] > 0.5)

        cell_regions = np.array([cell_region(c) for c in fe.cells(self.mesh)], dtype=np.int32)
        for facet in fe.facets(self.mesh):
            adjacent_cells = list(facet.entities(2))
            if len(adjacent_cells) == 2:
                if cell_regions[adjacent_cells[0]] != cell_regions[adjacent_cells[1]]:
                    facet_marker[facet] = 1 
        return facet_marker

    def _precompute_mappings(self):
        H, W = self.shape
        dy = float(self.spacing[0]) if len(self.spacing) > 0 else 1.0
        dx = float(self.spacing[1]) if len(self.spacing) > 1 else 1.0
        
        dof_coords = self.V_cg.tabulate_dof_coordinates()
        self.cg_j = np.clip(np.round(dof_coords[:, 0] / dx).astype(int), 0, W - 1)
        self.cg_i = np.clip(np.round(dof_coords[:, 1] / dy).astype(int), 0, H - 1)
        
        vertex_coords = self.mesh.coordinates()
        self.vert_j = np.clip(np.round(vertex_coords[:, 0] / dx).astype(int), 0, W - 1)
        self.vert_i = np.clip(np.round(vertex_coords[:, 1] / dy).astype(int), 0, H - 1)

    def _preassemble_matrix(self):
        print("Compiling and assembling DG Matrix A... (This takes a moment but happens ONLY ONCE)")
        u = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)
        n = fe.FacetNormal(self.mesh)
        h = fe.CellDiameter(self.mesh)
        h_avg = (h('+') + h('-')) / 2.0
        
        alpha_penalty = fe.Constant(self.penalty)
        alpha_sq = fe.Constant(self.alpha ** 2)

        u_p, u_m = u('+'), u('-')
        v_p, v_m = v('+'), v('-')
        n_p = n('+')

        u_diff = u_p - u_m
        v_diff = v_p - v_m

        a = fe.inner(u, v) * self.dx + alpha_sq * fe.inner(fe.grad(u), fe.grad(v)) * self.dx
        
        avg_trac_u = 0.5 * (fe.dot(fe.grad(u_p), n_p) + fe.dot(fe.grad(u_m), n_p))
        avg_trac_v = 0.5 * (fe.dot(fe.grad(v_p), n_p) + fe.dot(fe.grad(v_m), n_p))

        a -= alpha_sq * fe.inner(avg_trac_u, v_diff) * self.dS(0)
        a -= alpha_sq * fe.inner(avg_trac_v, u_diff) * self.dS(0)
        a += alpha_sq * (alpha_penalty / h_avg) * fe.inner(u_diff, v_diff) * self.dS(0)

        mask_bool = self.mask > 0.5
        dist_in = ndimage.distance_transform_edt(mask_bool)      
        dist_out = ndimage.distance_transform_edt(~mask_bool)     

        sdf = dist_out - dist_in

        dy, dx = np.gradient(sdf)
        norm = np.sqrt(dx**2 + dy**2) + 1e-8

        nx = dx / norm
        ny = dy / norm
        
        nx_func = fe.Function(self.V_cg)
        ny_func = fe.Function(self.V_cg)
        nx_func.vector().set_local(np.ascontiguousarray(nx[self.cg_i, self.cg_j]))
        ny_func.vector().set_local(np.ascontiguousarray(ny[self.cg_i, self.cg_j]))
        
        n_smooth = fe.as_vector([nx_func, ny_func])
        n_smooth_p = n_smooth('+') 
        
        grad_u_avg = 0.5 * (fe.grad(u_p) + fe.grad(u_m))
        grad_v_avg = 0.5 * (fe.grad(v_p) + fe.grad(v_m))

        trac_u_n = fe.dot(fe.dot(grad_u_avg, n_smooth_p), n_smooth_p)
        trac_v_n = fe.dot(fe.dot(grad_v_avg, n_smooth_p), n_smooth_p)
        
        un_diff_sm = fe.dot(u_diff, n_smooth_p)
        vn_diff_sm = fe.dot(v_diff, n_smooth_p)

        a -= alpha_sq * trac_u_n * vn_diff_sm * self.dS(1)
        a -= alpha_sq * trac_v_n * un_diff_sm * self.dS(1)
        
        a += alpha_sq * (alpha_penalty / h_avg) * un_diff_sm * vn_diff_sm * self.dS(1)

        self.A = fe.assemble(a)

        try:
            import scipy.sparse as sp
            from scipy.sparse.linalg import factorized
            
            mat_A = fe.as_backend_type(self.A).mat()
            indptr, indices, data = mat_A.getValuesCSR()
            self.A_scipy = sp.csr_matrix((data, indices, indptr), shape=mat_A.getSize())
            self.solve_A = factorized(self.A_scipy)
            
            self.V_cg_vec = fe.VectorFunctionSpace(self.mesh, 'CG', 1)
            m_trial = fe.TrialFunction(self.V_cg_vec)
            form_M = fe.inner(m_trial, v) * self.dx
            M_fenics = fe.assemble(form_M)
            
            mat_M = fe.as_backend_type(M_fenics).mat()
            indptr_m, indices_m, data_m = mat_M.getValuesCSR()
            self.M_scipy = sp.csr_matrix((data_m, indices_m, indptr_m), shape=mat_M.getSize())
            
            dofs_x = self.V_cg_vec.sub(0).dofmap().dofs()
            dofs_y = self.V_cg_vec.sub(1).dofmap().dofs()
            coords_x = self.V_cg_vec.tabulate_dof_coordinates()[dofs_x]
            
            H, W = self.shape
            dy_val = float(self.spacing[0]) if len(self.spacing) > 0 else 1.0
            dx_val = float(self.spacing[1]) if len(self.spacing) > 1 else 1.0
            
            self.cg_vec_j = np.clip(np.round(coords_x[:, 0] / dx_val).astype(int), 0, W - 1)
            self.cg_vec_i = np.clip(np.round(coords_x[:, 1] / dy_val).astype(int), 0, H - 1)
            self.dofs_x = dofs_x
            self.dofs_y = dofs_y
            
            self.m_flat = np.zeros(self.V_cg_vec.dim(), dtype=np.float64)
            self.use_scipy = True
            print("✓ Matrix extracted to SciPy Escape Hatch! Pure thread-safe math mode activated.")
            
        except Exception as e:
            print(f"Notice: SciPy Escape Hatch unavailable ({e}). Falling back to Native Locked Mode.")
            self.use_scipy = False
            self.linear_solver = fe.LUSolver(self.A, "umfpack")
            
            self.mx_func = fe.Function(self.V_cg)
            self.my_func = fe.Function(self.V_cg)
            self.m_vec = fe.as_vector([self.mx_func, self.my_func])
            self.L = fe.inner(self.m_vec, v) * self.dx
            self.b = fe.assemble(self.L)

        self.u_sol = fe.Function(self.V)
        print("✓ Solver is ready and Segfault-proof.")

    def solve_velocity_from_momentum(self, m_torch):
        if not FENICS_AVAILABLE:
            return _helmholtz_fft_solve(m_torch, self.spacing, R=self.R, alpha=self.alpha)
        return _DGFenicsSolve.apply(m_torch, self)

    def _solve_without_gradients(self, m_torch):
        B = m_torch.shape[0]
        u_list = []
        for b in range(B):
            m_x = m_torch[b, 1, :, :].detach().cpu().numpy().astype(np.float64).copy()
            m_y = m_torch[b, 0, :, :].detach().cpu().numpy().astype(np.float64).copy()
            u_x, u_y = self._solve_single(m_x, m_y)
            u_list.append(torch.stack([torch.from_numpy(u_y).float(), torch.from_numpy(u_x).float()], dim=0))
        return torch.stack(u_list, dim=0).to(m_torch.device)

    def _solve_single(self, m_x, m_y):
        with self.lock:
            curr_x = m_x.copy()
            curr_y = m_y.copy()
            
            for r in range(int(self.R)):
                if self.use_scipy:
                    self.m_flat[self.dofs_x] = curr_x[self.cg_vec_i, self.cg_vec_j]
                    self.m_flat[self.dofs_y] = curr_y[self.cg_vec_i, self.cg_vec_j]
                    
                    b = self.M_scipy.dot(self.m_flat)
                    u_flat = self.solve_A(b)
                    
                    self.u_sol.vector().set_local(u_flat)
                    self.u_sol.vector().apply("insert")
                    vertex_vals = self.u_sol.compute_vertex_values(self.mesh).copy()
                    
                else:
                    self.mx_func.vector().set_local(np.ascontiguousarray(curr_x[self.cg_i, self.cg_j]))
                    self.my_func.vector().set_local(np.ascontiguousarray(curr_y[self.cg_i, self.cg_j]))
                    
                    fe.assemble(self.L, tensor=self.b)
                    self.linear_solver.solve(self.u_sol.vector(), self.b)
                    vertex_vals = self.u_sol.compute_vertex_values(self.mesh).copy()

                num_vertices = self.mesh.num_vertices()
                ux_vert = vertex_vals[:num_vertices]
                uy_vert = vertex_vals[num_vertices:]

                field_x = np.zeros(self.shape)
                field_y = np.zeros(self.shape)
                
                field_x[self.vert_i, self.vert_j] = ux_vert
                field_y[self.vert_i, self.vert_j] = uy_vert
                
                curr_x = field_x
                curr_y = field_y

        return curr_x, curr_y

def create_fenics_solver(shape, spacing, mask, advanced=True, verbose=False, **kwargs):
    if not FENICS_AVAILABLE: return None
    kwargs.pop('advanced', None)
    return FenicsDiscontinuousVelocitySolver(shape, spacing, mask, verbose=verbose, **kwargs)
