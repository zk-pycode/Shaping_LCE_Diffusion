"""
Swelling_v20.1_GPU.py

GPU (CuPy) solver for diffusion-driven swelling of a liquid-crystal
elastomer disc. A finite solvent reservoir feeds a conservative FEM
diffusion field, and an Arruda-Boyce prism-FEM force kernel drives the
mechanical relaxation. Fields are written to HDF5 + XDMF for ParaView.

Model
------------------------------------
1. Connectivity
       Each cell is split by a center node into 4 prisms, linking both
       diagonals symmetrically (no single-diagonal bias).
2. Conservative diffusion
       Symmetric FEM stiffness K + lumped mass M give the update
       mu <- mu - dt * M^-1 (K @ mu), which conserves total solvent.
       Anisotropy enters as D = D_mu * diag(ax, ay, az).
3. Finite reservoir
       A fixed solvent volume is fed through the top-surface nodes under
       the droplet footprint via a clamp-and-deplete gate, then released.
4. Precision
       Arrays and the force kernel run in float32 (FEM operators are
       assembled in float64 then cast) for GPU throughput.

Output
------
    results/swelling_v20_gpu.h5      per-frame geometry and fields
    results/swelling_v20_gpu.xdmf    ParaView time-series wrapper
"""

import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpsp
from scipy import sparse as sp_cpu
import h5py
from progiter import ProgIter


# ---------------------------------------------------------------------------
# 3D prism mesh setup (deformed to elliptical domain)
# ---------------------------------------------------------------------------

nx, ny, nz = 32, 32, 3
lx, ly, lz = 6.0e-3, 6.0e-3, 0.1e-3

x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)
z = np.linspace(0, lz, nz)

xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
points_ref = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)
n_nodes = points_ref.shape[0]

print(f"Full grid: {n_nodes} nodes. Deforming to elliptical boundary...", flush=True)

dx = lx / (nx - 1)
dy = ly / (ny - 1)
dz = lz / (nz - 1)
cx, cy = lx / 2, ly / 2

def _idx(i, j, k):
    return i * ny * nz + j * nz + k

# ---------------------------------------------------------------------------
# Edge handling: drop poorly-cut cells, then project to the ellipse
# ---------------------------------------------------------------------------
# Edge handling (v15-style): keep cells with >= 2 corners inside the
# ellipse, THEN project. A cell with 0 or 1 corner inside would have
# 3+ of its nodes collapsed onto the boundary -> a badly folded prism
# (the intercardinal distortion). Dropping such cells up front means at
# most 2 corners of any kept cell ever move, bounding the distortion.
def _in_ellipse(i, j):
    return ((x[i] - cx)**2 / cx**2 + (y[j] - cy)**2 / cy**2) <= 1.0

keep_col = np.zeros((nx - 1, ny - 1), dtype=bool)
for i in range(nx - 1):
    for j in range(ny - 1):
        n_in = (int(_in_ellipse(i,   j  )) + int(_in_ellipse(i+1, j  )) +
                int(_in_ellipse(i+1, j+1)) + int(_in_ellipse(i,   j+1)))
        keep_col[i, j] = (n_in >= 2)
print(f"Cells: {(nx-1)*(ny-1)} total, {int(keep_col.sum())} kept "
      f"({(nx-1)*(ny-1)-int(keep_col.sum())} dropped: 0/1 corner inside)", flush=True)

# Nodes that belong to at least one kept cell
node_in_kept = np.zeros((nx, ny), dtype=bool)
for i in range(nx - 1):
    for j in range(ny - 1):
        if keep_col[i, j]:
            node_in_kept[i, j]     = node_in_kept[i+1, j]   = True
            node_in_kept[i+1, j+1] = node_in_kept[i, j+1]   = True

# Project only kept outside-nodes radially onto the ellipse boundary
n_proj = 0
for i in range(nx):
    for j in range(ny):
        if not node_in_kept[i, j]:
            continue
        rx, ry = x[i] - cx, y[j] - cy
        r_e = np.sqrt(rx*rx / cx**2 + ry*ry / cy**2)
        if r_e > 1.0:
            s = 1.0 / r_e
            px, py = cx + rx*s, cy + ry*s
            for k in range(nz):
                points_ref[_idx(i, j, k), 0] = px
                points_ref[_idx(i, j, k), 1] = py
            n_proj += 1
print(f"Projected {n_proj} XY grid points onto ellipse boundary.", flush=True)

# ---------------------------------------------------------------------------
# Add center nodes for kept cells only (v13-style: both diagonals)
# ---------------------------------------------------------------------------

#   One center node per (kept cell, z-level) at the mean of the 4 corner
#   nodes; each cell splits into 4 prisms fanning from the center, so
#   v0-v2 and v1-v3 are both linked (no single-diagonal bias).
n_nodes_orig = n_nodes
center_map   = {}                 # (i, j, k) -> absolute node index
center_list  = []
for i in range(nx - 1):
    for j in range(ny - 1):
        if not keep_col[i, j]:
            continue
        for k in range(nz):
            v0 = _idx(i, j, k);     v1 = _idx(i+1, j, k)
            v2 = _idx(i+1, j+1, k); v3 = _idx(i, j+1, k)
            center_map[(i, j, k)] = n_nodes_orig + len(center_list)
            center_list.append((points_ref[v0] + points_ref[v1] +
                                points_ref[v2] + points_ref[v3]) * 0.25)
if center_list:
    points_ref = np.vstack([points_ref, np.array(center_list, dtype=np.float64)])
n_nodes = points_ref.shape[0]
print(f"Center nodes added: {len(center_list)}. Total nodes: {n_nodes}", flush=True)

# ---------------------------------------------------------------------------
# Build prism connectivity (4 prisms per kept cell)
# ---------------------------------------------------------------------------

prism_list = []
for i in range(nx - 1):
    for j in range(ny - 1):
        if not keep_col[i, j]:
            continue
        for k in range(nz - 1):
            v0 = _idx(i,   j,   k);   v1 = _idx(i+1, j,   k)
            v2 = _idx(i+1, j+1, k);   v3 = _idx(i,   j+1, k)
            v4 = _idx(i,   j,   k+1); v5 = _idx(i+1, j,   k+1)
            v6 = _idx(i+1, j+1, k+1); v7 = _idx(i,   j+1, k+1)
            cb = center_map[(i, j, k)]; ct = center_map[(i, j, k+1)]
            prism_list.append([v0, v1, cb, v4, v5, ct])
            prism_list.append([v1, v2, cb, v5, v6, ct])
            prism_list.append([v2, v3, cb, v6, v7, ct])
            prism_list.append([v3, v0, cb, v7, v4, ct])

prism_cells = np.array(prism_list, dtype=np.int32)   # int32 matches CUDA C kernel

# ---------------------------------------------------------------------------
# Initialize fields / masks (computed on clean geometry)
# ---------------------------------------------------------------------------

x_coords = points_ref[:, 0]
y_coords = points_ref[:, 1]
z_coords = points_ref[:, 2]

mu_chem = np.zeros(n_nodes, dtype=np.float64)
phi     = np.zeros(n_nodes, dtype=np.float64)

bottom_boundary_mask = (z_coords < 1e-6)
print(f"Bottom boundary nodes: {np.sum(bottom_boundary_mask)}", flush=True)

# ---------------------------------------------------------------------------
# Reservoir source nodes under the droplet footprint
# ---------------------------------------------------------------------------
# Reservoir = every top-surface node under the droplet footprint (the droplet
# wets a continuous area, so both the 12 corner grid nodes AND the face-center
# nodes between them must be sources). The footprint radius R_drop is set by the
# 12 nearest ORIGINAL top grid nodes (the rounded 4x4-minus-corners patch); all
# top nodes (grid + center) within R_drop are then included -> 12 + 9 = 21 nodes.
n_source_corner = 12
top_mask = (z_coords > lz - 1e-6)                      # ALL top-surface nodes
top_idx  = np.where(top_mask)[0]
r_top    = np.sqrt((x_coords[top_idx] - cx)**2 + (y_coords[top_idx] - cy)**2)
orig_top = top_idx[top_idx < n_nodes_orig]            # original grid nodes on top
r_orig   = np.sort(np.sqrt((x_coords[orig_top] - cx)**2 + (y_coords[orig_top] - cy)**2))
R_drop   = r_orig[n_source_corner - 1]               # radius of the 12-corner footprint
res_nodes = top_idx[r_top <= R_drop + 1e-12]         # corners + face-centers under droplet
reservoir_mask = np.zeros(n_nodes, dtype=bool)
reservoir_mask[res_nodes] = True
print(f"Reservoir (source) nodes: {np.sum(reservoir_mask)}", flush=True)

# ---------------------------------------------------------------------------
# Geometric perturbation (buckling seed) - disabled
# ---------------------------------------------------------------------------

#   Disabled: commented out, not deleted, in case we need it back.
# print("Applying geometric perturbation...", flush=True)
# pert_mask = (np.abs(x_coords - lx/2) < lx/(nx-1)) & \
#             (np.abs(y_coords - ly/2) < ly/(ny-1))
# points_ref[pert_mask, 2] += lz * 0.5

# ---------------------------------------------------------------------------
# Filter degenerate elements (collapsed/folded by projection)
# ---------------------------------------------------------------------------

#   centroid Jacobian: xi=eta=1/3, zeta=0, zm=zp=0.5
#   The center-node split folds some boundary prisms inside-out (dJ < 0). Use a
#   SIGNED, volume-relative threshold so inverted folds and thin slivers are both
#   removed (an |dJ| test would keep the big inverted ones -> ugly edge "tabs").
dNxi_c = np.array([[-0.5,  0.5,  0.0, -0.5,  0.5,  0.0],
                   [-0.5,  0.0,  0.5, -0.5,  0.0,  0.5],
                   [-1/6, -1/6, -1/6,  1/6,  1/6,  1/6]])
ref_e_all   = points_ref[prism_cells]                      # (n_elem, 6, 3)
J_all       = np.einsum('ij,ejk->eik', dNxi_c, ref_e_all)  # (n_elem, 3, 3)
dJ_all      = np.linalg.det(J_all)
nominal_vol = (dx * dy * dz) / 4.0                          # ~ one interior prism
valid_mask  = dJ_all > 0.05 * nominal_vol                  # positively oriented, non-sliver
# Kernel degenerate/div-by-zero guard, scaled to the mesh so changing nx/ny/nz
# never silently skips all elements (a fixed absolute threshold would).
dJ_min_kernel = 1.0e-6 * nominal_vol
prism_cells = prism_cells[valid_mask]
n_elements  = len(prism_cells)
print(f"Mesh: {n_nodes} nodes, {n_elements} valid prism elements "
      f"(removed {np.sum(~valid_mask)} degenerate/folded)", flush=True)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

# --- Material (Arruda-Boyce, per-axis) ---
mu_0             = 1.0e+8   # shear modulus [Pa]
lam_0            = 1.0e+9   # bulk (Lame) modulus [Pa]
N_x              = 5.0      # chain segments along x (lower = stiffer stiffening onset)
N_y              = 5.0      # chain segments along y
N_z              = 5.0      # chain segments along z
softening_factor = 0.0      # phi-dependent softening of mu_0/lam_0

# --- Diffusion (conservative FEM, anisotropic) ---
D_mu             = 1.0e-9   # base chemical-potential diffusivity
diff_aniso_x     = 1.0e-0      # relative diffusivity along x (>1 = faster, <1 = slower)
diff_aniso_y     = 1.0e-0      # relative diffusivity along y
diff_aniso_z     = 1.0e-3      # relative diffusivity along z
k_phi            = 1.00    # mu_chem -> phi conversion factor

# --- Time integration ---
dt               = 2.5e-4
nsteps           = int(3600.0 / dt)
damping          = 0.990

# ---------------------------------------------------------------------------
# Finite solvent budget (droplet feeding the source nodes)
# ---------------------------------------------------------------------------

#   phi is a solvent volume fraction (J_sw = 1 + phi), so the solvent
#   volume held at a node is phi_i * v_i. The droplet releases a fixed
#   total volume V_solvent before the source switches off.
solvent_mass    = 0.50e-6          # kg     (0.5 mg of 5CB)
solvent_density = 1008.0          # kg/m^3 (5CB ~ 1.008 g/mL)
V_solvent       = solvent_mass / solvent_density   # m^3  (~4.96e-10 = 0.5 uL)
print(f"Solvent budget: {solvent_mass*1e6:.3f} mg -> V_solvent = {V_solvent*1e9:.4f} uL", flush=True)

# ---------------------------------------------------------------------------
# Output setup: HDF5 + XDMF file paths
# ---------------------------------------------------------------------------

h5_filename   = 'results/swelling_v20_gpu.h5'
xdmf_filename = 'results/swelling_v20_gpu.xdmf'
# XDMF must reference the HDF5 RELATIVE TO THE XDMF's own location. Both are
# written to the same directory, so use the basename (not the full path) - using
# the full 'results/...' path breaks ParaView when the xdmf is itself in results/.
h5_ref        = h5_filename.rsplit('/', 1)[-1]
frame_record  = int(1.0 / dt)            # 0.1 s per saved frame (3000 frames over 300 s)

# ---------------------------------------------------------------------------
# Conservative FEM diffusion operator
# ---------------------------------------------------------------------------

#   Assemble symmetric stiffness K_ab = int grad(N_a).D.grad(N_b) dV
#   and lumped mass (nodal volumes). Update:  mu -= dt * Minv (K @ mu)
#   Anisotropic diffusivity tensor D = D_mu * diag(ax, ay, 1).
print("Assembling conservative FEM Laplacian (K) and mass matrix (M)...", flush=True)
Dvec = D_mu * np.array([diff_aniso_x, diff_aniso_y, diff_aniso_z])

# Gauss points/weights (3-pt triangle x 2-pt line) - same scheme as force kernel
GX = np.array([1/3, 1/6, 1/6, 1/3, 1/3, 1/6, 1/6, 1/3])
GY = np.array([1/3, 1/6, 1/3, 1/6, 1/3, 1/6, 1/3, 1/6])
GZ = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * 0.5773502691896257
GW = np.array([1/3, 1/6, 1/6, 1/6, 1/3, 1/6, 1/6, 1/6])

ref_e_all = points_ref[prism_cells]                  # (ne, 6, 3)
ne        = n_elements
K_e       = np.zeros((ne, 6, 6))
elem_vol  = np.zeros(ne)

for g in range(8):
    xi, eta, zeta, w = GX[g], GY[g], GZ[g], GW[g]
    lam3 = 1.0 - xi - eta
    zm   = 0.5 * (1.0 - zeta)
    zp   = 0.5 * (1.0 + zeta)
    dNxi = np.array([
        [-zm,        zm,       0.0,      -zp,        zp,       0.0     ],   # d/dxi
        [-zm,        0.0,      zm,       -zp,        0.0,      zp      ],   # d/deta
        [-0.5*lam3, -0.5*xi,  -0.5*eta,   0.5*lam3,  0.5*xi,   0.5*eta ],   # d/dzeta
    ])                                               # (3, 6)
    J     = np.einsum('ij,ejk->eik', dNxi, ref_e_all)    # (ne, 3, 3)
    detJ  = np.abs(np.linalg.det(J))                      # (ne,) integration measure
    Jinv  = np.linalg.inv(J)                              # (ne, 3, 3)
    dNx   = np.einsum('eij,jk->eik', Jinv, dNxi)         # (ne, 3, 6) spatial grads
    K_e  += np.einsum('eia,i,eib->eab', dNx, Dvec, dNx) * (detJ * w)[:, None, None]
    elem_vol += detJ * w

# Scatter element matrices into the global sparse stiffness matrix
rows = np.repeat(prism_cells[:, :, None], 6, axis=2).ravel()
cols = np.repeat(prism_cells[:, None, :], 6, axis=1).ravel()
K_cpu = sp_cpu.coo_matrix((K_e.ravel(), (rows, cols)),
                          shape=(n_nodes, n_nodes)).tocsr()

# Lumped nodal volumes (mass matrix diagonal)
node_vol = np.zeros(n_nodes)
np.add.at(node_vol, prism_cells, (elem_vol / 6.0)[:, None])
Minv = 1.0 / np.maximum(node_vol, 1e-30)
print(f"Mesh volume: {node_vol.sum()*1e9:.4f} uL "
      f"(analytic disc ~ {np.pi*(lx/2)*(ly/2)*lz*1e9:.4f} uL)", flush=True)

# ---------------------------------------------------------------------------
# Raw CUDA C force kernel (Arruda-Boyce, prism FEM)
# ---------------------------------------------------------------------------

FORCE_KERNEL_CODE = r"""
extern "C" __global__ void force_kernel(
    const float* __restrict__ def_pos,    // (n_nodes*3,) row-major
    const float* __restrict__ ref_pos,    // (n_nodes*3,)
    const int*    __restrict__ cells,      // (n_elem*6,)
    float mu_0, float N_val, float lam_0, float sf,
    const float* __restrict__ phi_nodes,  // (n_nodes,)
    float* f_out,                         // (n_nodes*3,)  atomicAdd target
    int n_elem,
    float N_x, float N_y,                // per-axis chain segments (z uses N_val)
    float dJ_min                          // mesh-relative degenerate-skip / div-by-zero guard
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_elem) return;

    const float invNx = 1.0 / N_x;
    const float invNy = 1.0 / N_y;
    const float invNz = 1.0 / N_val;

    // Gauss point coordinates / weights
    const float GX[8] = {0.333333333333333,0.166666666666667,0.166666666666667,0.333333333333333,
                           0.333333333333333,0.166666666666667,0.166666666666667,0.333333333333333};
    const float GY[8] = {0.333333333333333,0.166666666666667,0.333333333333333,0.166666666666667,
                           0.333333333333333,0.166666666666667,0.333333333333333,0.166666666666667};
    const float GZ[8] = {-0.577350269189626,-0.577350269189626,-0.577350269189626,-0.577350269189626,
                            0.577350269189626, 0.577350269189626, 0.577350269189626, 0.577350269189626};
    const float GW[8] = {0.333333333333333,0.166666666666667,0.166666666666667,0.166666666666667,
                           0.333333333333333,0.166666666666667,0.166666666666667,0.166666666666667};

    // Connectivity and node positions
    int conn[6];
    float ref_e[6][3], def_e[6][3];
    for (int j = 0; j < 6; j++) {
        conn[j] = cells[e * 6 + j];
        ref_e[j][0] = ref_pos[conn[j]*3+0];
        ref_e[j][1] = ref_pos[conn[j]*3+1];
        ref_e[j][2] = ref_pos[conn[j]*3+2];
        def_e[j][0] = def_pos[conn[j]*3+0];
        def_e[j][1] = def_pos[conn[j]*3+1];
        def_e[j][2] = def_pos[conn[j]*3+2];
    }

    float ef[6][3];
    for (int j=0; j<6; j++) ef[j][0]=ef[j][1]=ef[j][2]=0.0;

    for (int g = 0; g < 8; g++) {
        float xi=GX[g], eta=GY[g], zeta=GZ[g], w=GW[g];
        float lam3=1.0-xi-eta, zm=0.5*(1.0-zeta), zp=0.5*(1.0+zeta);
        float Ns[6]={lam3*zm,xi*zm,eta*zm,lam3*zp,xi*zp,eta*zp};

        float phi_gp=0.0;
        for (int j=0; j<6; j++) phi_gp += Ns[j]*phi_nodes[conn[j]];

        float phi_c  = phi_gp<1.0 ? phi_gp : 1.0;
        float mu_loc = mu_0*(1.0-sf*phi_c);
        float lm_loc = lam_0*(1.0-sf*phi_c);
        float mu_h   = 0.5*mu_loc;
        float J_sw   = 1.0+phi_gp;
        float Jisq   = 1.0f/sqrtf(J_sw);

        float dNxi[3][6];
        dNxi[0][0]=-zm; dNxi[0][1]=zm;  dNxi[0][2]=0.0; dNxi[0][3]=-zp; dNxi[0][4]=zp;  dNxi[0][5]=0.0;
        dNxi[1][0]=-zm; dNxi[1][1]=0.0; dNxi[1][2]=zm;  dNxi[1][3]=-zp; dNxi[1][4]=0.0; dNxi[1][5]=zp;
        dNxi[2][0]=-0.5*lam3; dNxi[2][1]=-0.5*xi; dNxi[2][2]=-0.5*eta;
        dNxi[2][3]= 0.5*lam3; dNxi[2][4]= 0.5*xi; dNxi[2][5]= 0.5*eta;

        float J[3][3]={};
        for (int i=0;i<3;i++) for (int j=0;j<3;j++)
            for (int k=0;k<6;k++) J[i][j]+=dNxi[i][k]*ref_e[k][j];

        float dJ = J[0][0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1])
                  - J[0][1]*(J[1][0]*J[2][2]-J[1][2]*J[2][0])
                  + J[0][2]*(J[1][0]*J[2][1]-J[1][1]*J[2][0]);
        if (fabsf(dJ)<dJ_min) continue;   // dJ_min scales with element size (passed from host); real quality filtering is on the CPU (signed dJ > 0.05*nominal)

        float id=1.0/dJ, Ji[3][3];
        Ji[0][0]=(J[1][1]*J[2][2]-J[1][2]*J[2][1])*id; Ji[0][1]=(J[0][2]*J[2][1]-J[0][1]*J[2][2])*id; Ji[0][2]=(J[0][1]*J[1][2]-J[0][2]*J[1][1])*id;
        Ji[1][0]=(J[1][2]*J[2][0]-J[1][0]*J[2][2])*id; Ji[1][1]=(J[0][0]*J[2][2]-J[0][2]*J[2][0])*id; Ji[1][2]=(J[0][2]*J[1][0]-J[0][0]*J[1][2])*id;
        Ji[2][0]=(J[1][0]*J[2][1]-J[1][1]*J[2][0])*id; Ji[2][1]=(J[0][1]*J[2][0]-J[0][0]*J[2][1])*id; Ji[2][2]=(J[0][0]*J[1][1]-J[0][1]*J[1][0])*id;

        float dNx[3][6];
        for (int j=0;j<6;j++) for (int i=0;i<3;i++){
            dNx[i][j]=0; for(int k=0;k<3;k++) dNx[i][j]+=Ji[i][k]*dNxi[k][j]; }

        float Fm[3][3]={};
        for (int i=0;i<3;i++) for (int j=0;j<3;j++)
            for (int k=0;k<6;k++) Fm[i][j]+=def_e[k][i]*dNx[j][k];

        float Fe[3][3];
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) Fe[i][j]=Fm[i][j]*Jisq;

        float Je=Fe[0][0]*(Fe[1][1]*Fe[2][2]-Fe[1][2]*Fe[2][1])
                 -Fe[0][1]*(Fe[1][0]*Fe[2][2]-Fe[1][2]*Fe[2][0])
                 +Fe[0][2]*(Fe[1][0]*Fe[2][1]-Fe[1][1]*Fe[2][0]);
        if (Je<=0.0) continue;

        // Directional invariants: I4x/y/z = ||column of Fe||^2
        float I4x=0.0, I4y=0.0, I4z=0.0;
        for(int i=0;i<3;i++){I4x+=Fe[i][0]*Fe[i][0]; I4y+=Fe[i][1]*Fe[i][1]; I4z+=Fe[i][2]*Fe[i][2];}
        // Per-axis AB moduli: neo-Hookean base split equally + directional chain stiffening
        float dWx = mu_h/3.0 + 0.1*mu_loc*invNx*I4x + (11.0/350.0)*mu_loc*invNx*invNx*I4x*I4x;
        float dWy = mu_h/3.0 + 0.1*mu_loc*invNy*I4y + (11.0/350.0)*mu_loc*invNy*invNy*I4y*I4y;
        float dWz = mu_h/3.0 + 0.1*mu_loc*invNz*I4z + (11.0/350.0)*mu_loc*invNz*invNz*I4z*I4z;
        float cvol= lm_loc*logf(Je);
        float iJe =1.0/Je, Fei[3][3];
        Fei[0][0]=(Fe[1][1]*Fe[2][2]-Fe[1][2]*Fe[2][1])*iJe; Fei[0][1]=(Fe[0][2]*Fe[2][1]-Fe[0][1]*Fe[2][2])*iJe; Fei[0][2]=(Fe[0][1]*Fe[1][2]-Fe[0][2]*Fe[1][1])*iJe;
        Fei[1][0]=(Fe[1][2]*Fe[2][0]-Fe[1][0]*Fe[2][2])*iJe; Fei[1][1]=(Fe[0][0]*Fe[2][2]-Fe[0][2]*Fe[2][0])*iJe; Fei[1][2]=(Fe[0][2]*Fe[1][0]-Fe[0][0]*Fe[1][2])*iJe;
        Fei[2][0]=(Fe[1][0]*Fe[2][1]-Fe[1][1]*Fe[2][0])*iJe; Fei[2][1]=(Fe[0][1]*Fe[2][0]-Fe[0][0]*Fe[2][1])*iJe; Fei[2][2]=(Fe[0][0]*Fe[1][1]-Fe[0][1]*Fe[1][0])*iJe;

        float sq=sqrtf(J_sw), cv2=cvol*sq, P[3][3];
        for(int i=0;i<3;i++){
            P[i][0] = 2.0*sq*dWx*Fe[i][0] + cv2*Fei[0][i];
            P[i][1] = 2.0*sq*dWy*Fe[i][1] + cv2*Fei[1][i];
            P[i][2] = 2.0*sq*dWz*Fe[i][2] + cv2*Fei[2][i];
        }

        float dV=dJ*w;
        for (int j=0;j<6;j++){
            float p0=0,p1=0,p2=0;
            for(int k=0;k<3;k++){p0+=P[0][k]*dNx[k][j]; p1+=P[1][k]*dNx[k][j]; p2+=P[2][k]*dNx[k][j];}
            ef[j][0]-=p0*dV; ef[j][1]-=p1*dV; ef[j][2]-=p2*dV;
        }
    }

    for (int j=0; j<6; j++){
        int nd=conn[j];
        atomicAdd(&f_out[nd*3+0], ef[j][0]);
        atomicAdd(&f_out[nd*3+1], ef[j][1]);
        atomicAdd(&f_out[nd*3+2], ef[j][2]);
    }
}
"""

print("Compiling CUDA force kernel...", flush=True)
force_raw = cp.RawKernel(FORCE_KERNEL_CODE, 'force_kernel')
print("Force kernel compiled OK.", flush=True)

# ---------------------------------------------------------------------------
# Transfer arrays to GPU
# ---------------------------------------------------------------------------

print("Transferring data to GPU (float32 for 3090 throughput)...", flush=True)
points_ref_gpu  = cp.asarray(points_ref, dtype=cp.float32)         # (n_nodes, 3)
ref_pos_flat    = cp.asarray(points_ref.ravel(), dtype=cp.float32) # (n_nodes*3,)
prism_cells_gpu = cp.asarray(prism_cells.ravel())                 # (n_elem*6,)  int32
mu_chem_gpu     = cp.asarray(mu_chem, dtype=cp.float32)           # (n_nodes,)
phi_gpu         = cp.asarray(phi, dtype=cp.float32)               # (n_nodes,)
u_gpu           = cp.zeros((n_nodes, 3), dtype=cp.float32)
v_gpu           = cp.zeros((n_nodes, 3), dtype=cp.float32)
f_out_gpu       = cp.zeros(n_nodes * 3, dtype=cp.float32)
reservoir_gpu   = cp.asarray(reservoir_mask)

# Conservative diffusion operators on GPU (assembled in float64, stored float32)
K_gpu        = cpsp.csr_matrix(K_cpu.astype(np.float32))
node_vol_gpu = cp.asarray(node_vol, dtype=cp.float32)
Minv_gpu     = cp.asarray(Minv, dtype=cp.float32)
res_idx      = cp.asarray(np.where(reservoir_mask)[0])

# Saturate the source nodes at t=0; this initial charge counts against budget.
mu_chem_gpu[res_idx] = 1.0
injected = cp.asarray(np.float32(k_phi * node_vol[res_nodes].sum()))   # m^3 released so far
print("GPU transfer complete.", flush=True)

TPB      = 256
BPG_elem = (n_elements + TPB - 1) // TPB

# ---------------------------------------------------------------------------
# Output setup: open HDF5, write mesh topology
# ---------------------------------------------------------------------------

h5file = h5py.File(h5_filename, 'w')
h5file.create_dataset('Mesh/Topology', data=prism_cells.astype(np.int64), compression='gzip')
xdmf_steps = []

# ---------------------------------------------------------------------------
# Simulation loop (GPU via CuPy)
# ---------------------------------------------------------------------------

print(f"Running on GPU: {n_nodes} nodes, {n_elements} elements, {nsteps} steps...", flush=True)
pbar      = ProgIter(range(nsteps), desc="Computing", verbose=1)
frame_idx = 0

for step in pbar:
    # 1. Chemical potential diffusion - CONSERVATIVE: mu -= dt * Minv (K @ mu)
    mu_chem_gpu = mu_chem_gpu - dt * Minv_gpu * (K_gpu @ mu_chem_gpu)

    # 2. Finite-reservoir clamp-and-deplete gate (branchless, no host sync).
    #    While budget remains, top the 9 nodes back up toward mu=1 and bill the
    #    injected solvent volume; once spent (frac->0) the nodes diffuse freely.
    need      = cp.maximum(1.0 - mu_chem_gpu[res_idx], 0.0)
    inj_full  = k_phi * cp.sum(node_vol_gpu[res_idx] * need)        # vol to fully refill
    remaining = cp.maximum(V_solvent - injected, 0.0)
    frac      = cp.where(inj_full > 1e-30,
                         cp.clip(remaining / inj_full, 0.0, 1.0), 0.0)
    mu_chem_gpu[res_idx] += frac * need
    injected = injected + frac * inj_full

    # 3. Phi update
    phi_gpu = cp.maximum(k_phi * mu_chem_gpu, 0.0)

    # 4. Deformed positions (flat, for force kernel)
    def_pos_flat = (points_ref_gpu + u_gpu).ravel()

    # 5. Zero force array, then compute forces
    f_out_gpu[:] = 0.0
    force_raw(
        (BPG_elem,), (TPB,),
        (def_pos_flat, ref_pos_flat, prism_cells_gpu,
         np.float32(mu_0), np.float32(N_z), np.float32(lam_0),
         np.float32(softening_factor), phi_gpu,
         f_out_gpu, np.int32(n_elements),
         np.float32(N_x), np.float32(N_y),
         np.float32(dJ_min_kernel))
    )
    f_int_gpu = f_out_gpu.reshape(n_nodes, 3)

    # 6. Velocity update + global damping + rigid-body drift removal (floating sample)
    v_gpu += dt * f_int_gpu
    v_gpu *= damping
    v_gpu -= cp.mean(v_gpu, axis=0, keepdims=True)

    # 7. Position update
    u_gpu += dt * v_gpu

    # 8. Reflective floor at z=0
    new_z     = points_ref_gpu[:, 2] + u_gpu[:, 2]
    violating = new_z < 0.0
    u_gpu[violating, 2] = -points_ref_gpu[violating, 2]
    v_gpu[violating, 2] = -v_gpu[violating, 2]

    # 9. Frame output (GPU → CPU only on save steps)
    if step % frame_record == 0:
        cp.cuda.Stream.null.synchronize()
        positions_h = (points_ref_gpu + u_gpu).get()
        mu_chem_h   = mu_chem_gpu.get()
        phi_h       = phi_gpu.get()
        disp_mag    = cp.linalg.norm(u_gpu, axis=1).get()
        inj_now     = float(injected.get())

        grp_name = f'Step_{frame_idx:06d}'
        grp = h5file.create_group(grp_name)
        grp.create_dataset('Geometry',               data=positions_h, compression='gzip')
        grp.create_dataset('mu_chem',                data=mu_chem_h,   compression='gzip')
        grp.create_dataset('phi',                    data=phi_h,       compression='gzip')
        grp.create_dataset('displacement_magnitude', data=disp_mag,    compression='gzip')
        grp.attrs['solvent_injected_uL'] = inj_now * 1e9
        grp.attrs['time'] = step * dt

        xdmf_steps.append({'grp': grp_name, 'time': step * dt})
        frame_idx += 1

h5file.close()
print(f"HDF5 written: {h5_filename}  ({frame_idx} frames). "
      f"Solvent injected: {float(injected.get())*1e9:.4f} / {V_solvent*1e9:.4f} uL", flush=True)

# ---------------------------------------------------------------------------
# Write XDMF file
# ---------------------------------------------------------------------------

print(f"Writing XDMF: {xdmf_filename} ...", flush=True)

def xdmf_dataitem(path, dims, dtype='Float', precision=4, fmt='HDF'):
    """Return an XDMF <DataItem> element pointing at an HDF5 path.

    Parameters
    ----------
    path : str
        HDF5 dataset path (referenced relative to the XDMF via h5_ref).
    dims : sequence of int
        Dataset dimensions.
    dtype : str, optional
        XDMF data type ('Float' or 'Int').
    precision : int, optional
        Byte precision of the stored data.
    fmt : str, optional
        Storage format ('HDF').
    """
    dim_str = ' '.join(str(d) for d in dims)
    return (f'          <DataItem Format="{fmt}" DataType="{dtype}" '
            f'Precision="{precision}" Dimensions="{dim_str}">\n'
            f'            {h5_ref}:{path}\n'
            f'          </DataItem>\n')

lines = []
lines.append('<?xml version="1.0" ?>')
lines.append('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>')
lines.append('<Xdmf Version="3.0">')
lines.append('  <Domain>')
lines.append('    <Grid GridType="Collection" CollectionType="Temporal" Name="TimeSeries">')

for s in xdmf_steps:
    grp = s['grp']
    t   = s['time']
    lines.append(f'      <Grid Name="{grp}" GridType="Uniform">')
    lines.append(f'        <Time Value="{t:.6f}"/>')

    lines.append(f'        <Topology TopologyType="Wedge" NumberOfElements="{n_elements}">')
    lines.append(xdmf_dataitem('/Mesh/Topology', [n_elements, 6], dtype='Int', precision=8).rstrip())
    lines.append('        </Topology>')

    lines.append('        <Geometry GeometryType="XYZ">')
    lines.append(xdmf_dataitem(f'/{grp}/Geometry', [n_nodes, 3]).rstrip())
    lines.append('        </Geometry>')

    for field_name, ds_name in [
        ('Chemical Potential',          'mu_chem'),
        ('Solvent Concentration (phi)', 'phi'),
        ('Displacement Magnitude',      'displacement_magnitude'),
    ]:
        lines.append(f'        <Attribute Name="{field_name}" AttributeType="Scalar" Center="Node">')
        lines.append(xdmf_dataitem(f'/{grp}/{ds_name}', [n_nodes]).rstrip())
        lines.append('        </Attribute>')

    lines.append('      </Grid>')

lines.append('    </Grid>')
lines.append('  </Domain>')
lines.append('</Xdmf>')

with open(xdmf_filename, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f"Done. Open '{xdmf_filename}' in ParaView (File > Open, select XDMF reader).", flush=True)
