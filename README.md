
# WENO

##Includes
- Classical reconstructions: WENO‑JS/Z, ENO3, TVD‑minmod, PPM, Lagrange5
- Scalar PDEs: 3D advection (exact-ref available), advection–diffusion
- System PDE: 2D rotating shallow water (f‑plane)
- Data generation for smooth + discontinuous fields
- Training scaffold for a gated MLP (weights + ENO gate)

See `scripts/make_videos.py` or run:
```bash
python -m wenoml.scripts.make_videos --case advection3d_slice --out figures/videos/my_adv.gif --frames 60 --duration 0.6
python -m wenoml.scripts.make_videos --case shallow_water_2d   --out figures/videos/my_sw.gif  --frames 80 --duration 1.0
```
