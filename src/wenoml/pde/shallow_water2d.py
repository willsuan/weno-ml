
import numpy as np
from ..weno5 import weno5_reconstruct
from ..classical import eno_reconstruct, tvd_minmod_reconstruct, ppm_reconstruct
def flux(U, g=9.81, axis=0):
    h,hu,hv = U[...,0], U[...,1], U[...,2]
    u = np.where(h>1e-12, hu/h, 0.0); v = np.where(h>1e-12, hv/h, 0.0)
    if axis==0: return np.stack([hu, hu*u + 0.5*g*h*h, hu*v], -1)
    else:       return np.stack([hv, hv*u, hv*v + 0.5*g*h*h], -1)
def reconstruct(U, method='weno_z', axis=0):
    comps=[]
    for c in range(U.shape[-1]):
        if method=='weno_z': ul,ur=weno5_reconstruct(U[...,c], axis=axis, mode='Z')
        elif method=='eno':  ul,ur=eno_reconstruct(U[...,c], axis=axis)
        elif method=='tvd_minmod': ul,ur=tvd_minmod_reconstruct(U[...,c], axis=axis)
        elif method=='ppm':  ul,ur=ppm_reconstruct(U[...,c], axis=axis)
        else: raise NotImplementedError(method)
        comps.append((ul,ur))
    UL=np.stack([c[0] for c in comps], -1); UR=np.stack([c[1] for c in comps], -1)
    return UL,UR
def max_speed(U, g=9.81):
    h,hu,hv = U[...,0], U[...,1], U[...,2]
    u = np.where(h>1e-12, hu/h, 0.0); v = np.where(h>1e-12, hv/h, 0.0)
    c = np.sqrt(g*np.maximum(h, 0.0))
    return float(np.max(np.abs(u)+c)+np.max(np.abs(v)+c))
def rusanov_flux(UL, UR, axis=0, g=9.81, fcor=0.0):
    FL = flux(UL, g=g, axis=axis); FR = flux(UR, g=g, axis=axis)
    hL,huL,hvL = UL[...,0],UL[...,1],UL[...,2]; hR,huR,hvR = UR[...,0],UR[...,1],UR[...,2]
    uL = np.where(hL>1e-12, huL/hL, 0.0); vL = np.where(hL>1e-12, hvL/hL, 0.0)
    uR = np.where(hR>1e-12, huR/hR, 0.0); vR = np.where(hR>1e-12, hvR/hR, 0.0)
    cL = np.sqrt(g*np.maximum(hL,0.0)); cR = np.sqrt(g*np.maximum(hR,0.0))
    s = np.maximum(np.abs(uL)+cL, np.abs(uR)+cR) if axis==0 else np.maximum(np.abs(vL)+cL, np.abs(vR)+cR)
    return 0.5*(FL+FR) - 0.5*s[...,None]*(UR-UL)
def step_shallow_water(h, hu, hv, g=9.81, fcor=0.0, method='weno_z', cfl=0.4, t_end=0.5):
    U=np.stack([h,hu,hv], -1)
    def rhs(curr):
        ULx,URx = reconstruct(curr, method=method, axis=0); Fx=rusanov_flux(ULx,URx,axis=0,g=g,fcor=fcor)
        ULy,URy = reconstruct(curr, method=method, axis=1); Fy=rusanov_flux(ULy,URy,axis=1,g=g,fcor=fcor)
        div=(np.roll(Fx,-1,0)-Fx)+(np.roll(Fy,-1,1)-Fy)
        h,hu,hv = curr[...,0],curr[...,1],curr[...,2]
        S = np.stack([np.zeros_like(h), fcor*(-hv), fcor*(hu)], -1)
        return -(div - S)
    dt = cfl / max(1e-12, max_speed(U,g=g)); t=0.0
    while t<t_end-1e-12:
        if t+dt>t_end: dt=t_end-t
        U1=U + dt*rhs(U)
        U2=0.75*U + 0.25*(U1 + dt*rhs(U1))
        U = (1.0/3.0)*U + (2.0/3.0)*(U2 + dt*rhs(U2)); t+=dt
        dt = cfl / max(1e-12, max_speed(U,g=g))
    return U[...,0], U[...,1], U[...,2]
