# WENO-ML

## Overview

The WENO-ML project develops a machine-learning framework to learn optimal nonlinear reconstruction weights for high-order finite-volume and finite-difference solvers of nonlinear PDEs that exhibit smooth and discontinuous features—such as shocks, contacts, and filaments.

It unites the analytical rigor of Weighted Essentially Non-Oscillatory (WENO) schemes with modern data-driven regression, aiming for:
	•	Spectral-like accuracy in smooth regions,
	•	Non-oscillatory stability near discontinuities,
	•	Generalization across different equations (advection, Euler, MHD, etc.), and
	•	Interpretability grounded in classical numerical analysis.

The codebase provides both classical solvers (WENO-JS/Z, ENO3, TVD-minmod, PPM, Lagrange5) and learned reconstructions for scalar and system PDEs, including:
	•	3D Advection and Advection–Diffusion
	•	3D Burgers
	•	3D Compressible Euler (Characteristic WENO-Z, HLLC/Rusanov)
	•	2D Rotating Shallow-Water Equations (f-plane dynamics)

Thank you to Sergey Fomel for supervising this project.

⸻

## Mathematical Theory

### Weighted Essentially Non-Oscillatory (WENO) Reconstruction

The WENO family of schemes (Liu et al. 1994; Jiang & Shu 1996) reconstructs interface values u_{i+\frac12}^\pm using multiple sub-stencils and smoothness-adaptive weights.

For a uniform 1D grid, with a 5-point stencil S_i = \{u_{i-2}, u_{i-1}, u_i, u_{i+1}, u_{i+2}\}, define three third-order polynomials:
q_0 = \tfrac16(2u_{i-2} - 7u_{i-1} + 11u_i),\quad
q_1 = \tfrac16(-u_{i-1} + 5u_i + 2u_{i+1}),\quad
q_2 = \tfrac16(2u_i + 5u_{i+1} - u_{i+2}).

Each candidate’s smoothness is quantified by the Jiang–Shu indicators:
$\beta_k = \tfrac{13}{12}(u_{i+k-2} - 2u_{i+k-1} + u_{i+k})^2 + \tfrac14(u_{i+k-2} - 4u_{i+k-1} + 3u_{i+k})^2,$
and the nonlinear weights are:
$\alpha_k = \frac{d_k}{(\epsilon + \beta_k)^p},\qquad$
$w_k = \frac{\alpha_k}{\sum_j \alpha_j},$
where $d_k$ are optimal linear weights (for 5th-order accuracy, $d_0=0.1,\,d_1=0.6,\,d_2=0.3$).

The final left-state reconstruction:
$$u_{i+\frac12}^- = \sum_{k=0}^2 w_k q_k.$$

WENO-Z Improvement

Borges et al. (2008) proposed a global smoothness sensor
$$\tau_5 = |\beta_2 - \beta_0|,\quad$$
$$\alpha_k = d_k \Bigl[1 + \bigl(\tfrac{\tau_5}{\beta_k+\epsilon}\bigr)^p\Bigr]$$,
which restores optimal accuracy near critical points and reduces dissipation.

⸻

### Machine-Learned Weights

In this project, the nonlinear mapping $S_i \mapsto (w_0,w_1,w_2)$ is replaced by a neural functional:
$$(w_0, w_1, w_2, g) = f_\theta(u_{i-2}, u_{i-1}, u_i, u_{i+1}, u_{i+2})$$,
where $g \in [0,1]$ is a gating variable interpolating between learned WENO-like behavior and ENO-like limiting:
$$u^-{i+\frac12} = (1-g)\sum_k w_k q_k + g\,q{\mathrm{ENO}}$$.

The model is trained via robust regression and regularization losses:
$$\mathcal{L} =
\mathcal{L}{\text{Huber}} +
\lambda_Z \|\mathbf{w}-\mathbf{w}{\text{WENO-Z}}\|^2 +
\lambda_m \,\mathcal{P}_{\text{mono}} +
\lambda_g \,\mathrm{BCE}(g, \text{edge\flag}),
$$
where $\mathcal{P}{\text{mono}}$ penalizes violations of monotonicity, ensuring non-oscillatory reconstructions.

⸻

### Systems of Conservation Laws

For vector equations $\partial_t U + \nabla\cdot F(U) = 0,$
WENO reconstruction occurs in characteristic space:

$$W = L(U_L,U_R)\,U,\qquad$$
$$U = R(U_L,U_R)\,W,$$
where $L$ and $R$ are the left/right eigenvector matrices of the flux Jacobian.
Characteristic variables are reconstructed independently, and fluxes are obtained via approximate Riemann solvers such as HLLC or Rusanov.

Time integration uses a third-order TVD Runge–Kutta method.

⸻

### PDEs Included

PDE	Description	Numerical flux	Notes
Linear Advection	Passive scalar transport	Lax–Friedrichs	Exact reference via spectral shift
Burgers	Nonlinear scalar convection	Central + viscosity	Benchmarks shock formation
Advection–Diffusion	Constant velocity + Laplacian diffusion	Lax–Friedrichs + centered Laplacian	Tests stiffness & smoothness
Compressible Euler 3D	Conservation of mass, momentum, energy	HLLC / Rusanov	Characteristic WENO-Z reconstruction
Rotating Shallow-Water 2D	f-plane equations	Rusanov	Coriolis source terms, dam-break test


⸻

### Theoretical Significance

WENO schemes are the archetype of nonlinear adaptivity in numerical analysis—balancing accuracy and stability without explicit switch functions.
This project interprets the WENO mechanism as a learnable prior, where a neural network can approximate or improve the mapping from local field morphology to interpolation weights.

It extends the philosophy of physics-informed machine learning (PINNs, operator learning) to numerical discretization itself, enabling:
	•	Adaptive bias across regimes (laminar–turbulent, smooth–shock),
	•	Reduced need for hand-tuned regularization parameters,
	•	Better extensibility to new PDEs.

⸻

References
	1.	Liu, X.-D., Osher, S., & Chan, T. (1994). Weighted Essentially Non-Oscillatory Schemes. Journal of Computational Physics, 115, 200–212.
	2.	Jiang, G.-S., & Shu, C.-W. (1996). Efficient Implementation of Weighted ENO Schemes. Journal of Computational Physics, 126, 202–228.
	3.	Borges, R., Carmona, M., Costa, B., & Don, W. S. (2008). An Improved Weighted Essentially Non-Oscillatory Scheme for Hyperbolic Conservation Laws. Journal of Computational Physics, 227, 3191–3211.
	4.	Shu, C.-W. (2009). High Order Weighted Essentially Non-Oscillatory Schemes for Convection Dominated Problems. SIAM Review, 51, 82–126.
	5.	Toro, E. F. (2009). Riemann Solvers and Numerical Methods for Fluid Dynamics (3rd ed.). Springer.
	6.	Godunov, S. K. (1959). A Difference Scheme for Numerical Solution of Discontinuous Solution of Hydrodynamic Equations. Matematicheskii Sbornik, 47, 271–306.
	7.	van Leer, B. (1977). Towards the Ultimate Conservative Difference Scheme. IV. A New Approach to Numerical Convection. Journal of Computational Physics, 23, 276–299.
	8.	LeVeque, R. J. (2002). Finite Volume Methods for Hyperbolic Problems. Cambridge University Press.
	9.	Zhang, X., & Shu, C.-W. (2010). On Positivity-Preserving High Order Discontinuous Galerkin Schemes for Compressible Euler Equations. Journal of Computational Physics, 229, 8918–8934.
	10.	Mishra, S., & Ray, D. (2020). Physics-Informed Machine Learning for PDEs: Concepts, Algorithms, and Applications. Acta Numerica, 29, 1–131.

⸻

Next steps:
- Extend function generation and 3d test function set
- Train with different architectures
- More comprehensive benchmarks
- Adapt software for compute cluster


3D Linear Advection — Central Slice Evolution
A central x-slice of a passive scalar field advected with constant velocity.
The color field shows the transport of smooth and filamentary structures using a fifth-order WENO-Z scheme and TVD-RK3 time integration. Periodic boundaries preserve spectral features without diffusion.

![advection3d_slice_demo](https://github.com/user-attachments/assets/2bff4f2b-df6d-44eb-99ce-231f56dab208)


Rotating Shallow-Water Dam Break
Time evolution of the free-surface height $h(x,y,t)$ under rotation on an f-plane.
A step in initial water depth produces gravity waves and geostrophic adjustment. Computed with a 2D WENO-Z reconstruction and Rusanov flux; color represents normalized height.

![shallow_water_demo](https://github.com/user-attachments/assets/b49188e9-4cc7-46f8-a00b-6df8bf696d7b)
