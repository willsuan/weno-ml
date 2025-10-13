# WENO

Current capabilities
- Classical reconstructions: WENO‑JS/Z, ENO3, TVD‑minmod, PPM, Lagrange5
- Scalar PDEs: 3D advection (exact-ref available), advection–diffusion
- System PDE: 2D rotating shallow water (f‑plane)
- Data generation for smooth + discontinuous fields
- Training scaffold for a gated MLP (weights + ENO gate)


Next steps:
- Extend function generation and 3d test function set
- Train with different architectures
- Comprehensive benchmark
- Adapt software for compute cluster

Project currently supervised by Sergey Fomel.


![advection3d_slice_demo](https://github.com/user-attachments/assets/2bff4f2b-df6d-44eb-99ce-231f56dab208)
