## Diffusion-Driven Swelling and Shape Morphing of Liquid Crystal Elastomer Discs
<a href="https://doi.org/"><img alt="Static Badge" src="https://img.shields.io/badge/DOI-pending-yellow">
<a href="https://doi.org/"><img alt="Static Badge" src="https://img.shields.io/badge/DOI-pending-blue">

### Project Description

This project models the solvent driven swelling and out-of-plane shape morphing of a thin liquid crystal elastomer (LCE) disc immersed in a low-molecular-weight liquid crystal solvent. As the solvent diffuses into the anisotropic polymer network, the disc absorbs volume non-uniformly and buckles into a dome. The equilibrium and transient shapes are obtained by coupling a conservative finite-element diffusion solver to an anisotropic (per-axis Arruda–Boyce) hyperelastic force model on a prism mesh of the disc.

The disc floats freely (no clamped boundary) above a reflective floor at z = 0, so it is free to dome upward as the solvent enters. The chemical potential field is evolved with a symmetric FEM stiffness operator that conserves the total absorbed solvent, so a finite droplet reservoir is physically meaningful: a fixed solvent budget is fed in through the wetted top-surface nodes via a clamp-and-deplete gate, and the dome rises then flattens as the budget is spent. The mechanical response uses a directional chain-stiffening model (different chain-segment counts along x, y, z) to capture the nematic anisotropy of the LCE.

```
Project_Structure/
├── Swelling_v20.1_CPU.py       Conservative FEM diffusion, vectorized NumPy force kernel (CPU-only fork)
├── Swelling_v20.1_GPU.py       Identical physics + CUDA Arruda-Boyce force kernel (CuPy / GPU)
└── results/                    Output directory (HDF5 + XDMF written here; create if absent)
```

Both scripts are single-file and self-contained, and write an HDF5 + XDMF time-series of the deformed geometry, chemical potential, solvent concentration, and displacement magnitude - directly viewable in ParaView and compared against measured disc-height-vs-time data for quantitative validation.

Internally each script performs the same pipeline:

```
1. Build a prism mesh of the disc  -> elliptical projection, center-node split (4 prisms/cell)
2. Filter degenerate/folded elements (signed, mesh-relative Jacobian threshold)
3. Assemble conservative FEM diffusion operators  -> stiffness K + lumped mass M
4. Identify the droplet footprint (reservoir source nodes) on the top surface
5. Time loop:  diffuse mu  ->  reservoir clamp-and-deplete gate  ->  phi  ->  force  ->  velocity/position update  ->  reflective floor
6. Write HDF5 frames + an XDMF wrapper for ParaView
```

### Physics

Conservative diffusion. The chemical potential 'mu' is evolved by an explicit, solvent-conserving update

```
mu <- mu - dt * M^-1 (K @ mu)
```

where 'K' is the symmetric FEM stiffness matrix assembled from the prism shape-function gradients with an anisotropic diffusivity tensor 'D = D_mu * diag(ax, ay, az)', and 'M' is the lumped (diagonal) mass matrix of nodal volumes. Because '1^T K = 0', the total absorbed solvent 'sum_i v_i * phi_i' is conserved, which makes the finite reservoir meaningful.

Finite reservoir. A fixed solvent volume 'V_solvent = solvent_mass / solvent_density' is injected only through the top-surface nodes under the droplet footprint. A branchless clamp-and-deplete gate holds those nodes at 'mu = 1' while cumulative injected volume is below budget, then releases them to diffuse freely once the budget is spent (dome rises, then flattens).

Anisotropic hyperelasticity. The solvent concentration 'phi = k_phi * mu' sets a swelling stretch 'J_sw = 1 + phi'. The force kernel evaluates a per-axis Arruda–Boyce strain energy: a neo-Hookean base split equally across axes plus directional chain-stiffening controlled by independent chain-segment counts 'N_x, N_y, N_z', together with a volumetric 'lambda * log(J)' term. The first Piola–Kirchhoff stress is integrated over each prism (8-point Gauss quadrature) and scattered to nodal forces.

Dynamics. Nodal forces drive an explicit, over-damped velocity update with global damping and rigid-body drift removal (the sample floats). A reflective floor reverses any node that would cross 'z = 0'.

### Environment

| Package | Purpose |
|---------|---------|
| 'numpy' | Mesh construction, FEM assembly, CPU force kernel |
| 'cupy' + 'cupyx.scipy.sparse' | GPU arrays and sparse stiffness matrix (GPU version only) |
| 'scipy' | Sparse (COO/CSR) stiffness matrix assembly |
| 'h5py' | HDF5 time-series output |
| 'progiter' | Progress reporting |

The GPU script additionally requires a CUDA-capable GPU with a working CuPy/CUDA toolkit (the force kernel is JIT-compiled via 'cp.RawKernel'). The CPU script has no GPU dependency — it aliases 'cp = np' and runs entirely in NumPy.

### Running the Simulation

1. Create the output directory (both scripts write to 'results/'):

2. Edit parameters near the top of the script — mesh resolution, material moduli, anisotropy, diffusivity, solvent budget, and time step (see below).

3. Run:
```
python Swelling_v20.1_GPU.py     # GPU (recommended)
# or
python Swelling_v20.1_CPU.py     # CPU-only fork, identical physics
```

Progress, mesh statistics, the solvent budget, and total injected volume are printed to stdout.

### Key Parameters

| Parameter | Description | Value |
|-----------|-------------|-------|
| 'nx, ny, nz' | Mesh resolution (x, y, z grid nodes) | '32, 32, 3' |
| 'lx, ly, lz' | Disc dimensions (m) | '6e-3, 6e-3, 0.1e-3' |
| 'mu_0' | Shear modulus (Pa) | '1e8' |
| 'lam_0' | Bulk (Lamé) modulus (Pa) | '1e9' |
| 'N_x, N_y, N_z' | Arruda–Boyce chain segments per axis (lower = stiffer stiffening) | '35, 5, 5' |
| 'softening_factor' | phi-dependent softening of the moduli | '0.0' |
| 'D_mu' | Base chemical-potential diffusivity (m²/s) | '1e-9' |
| 'diff_aniso_x/y/z' | Relative diffusivity along each axis | '0.1, 0.1, 1e-3' |
| 'k_phi' | Chemical potential → solvent volume fraction conversion | '1.114' |
| 'dt' | Time step (s) | '2.5e-4' |
| 'nsteps' | Number of steps ('9000 / dt') | '3.6e7' |
| 'damping' | Global velocity damping per step | '0.990' |
| 'solvent_mass' | Total solvent budget (kg) | '0.25e-6' |
| 'solvent_density' | Solvent density, 5CB (kg/m³) | '1008.0' |
| 'frame_record' | Steps per saved frame ('1 / dt' → 1 s/frame) | '4000' |

### Output

```
results/
├── swelling_v20P2.xdmf          Time-series wrapper for ParaView (GPU run)
├── swelling_v20P2.h5            HDF5 data store (geometry, mu_chem, phi, |displacement|)
├── swelling_v20P2_cpu.xdmf      Time-series wrapper (CPU run)
└── swelling_v20P2_cpu.h5        HDF5 data store (CPU run)
```

Each saved frame stores the deformed nodal geometry, the chemical potential field, the solvent concentration 'phi', and the displacement magnitude, plus per-frame attributes for simulation time and cumulative injected solvent volume (µL). The mesh topology ('Wedge'/prism elements) is written once.

To visualise, open the '.xdmf' file in ParaView (File → Open → select the XDMF reader) and color by *Solvent Concentration (phi)* or *Displacement Magnitude*, or warp by geometry to see the dome grow over time.

---

### References

```
TBA

```
