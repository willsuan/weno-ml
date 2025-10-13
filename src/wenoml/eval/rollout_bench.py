
import numpy as np
from ..weno5 import weno5_reconstruct
from ..classical import eno_reconstruct, tvd_minmod_reconstruct, lagrange5_interface, ppm_reconstruct
def spectral_shift(u0, vel, t):
    nx,ny,nz=u0.shape
    U=np.fft.fftn(u0); kx=np.fft.fftfreq(nx)*nx; ky=np.fft.fftfreq(ny)*ny; kz=np.fft.fftfreq(nz)*nz
    kkx,kky,kkz=np.meshgrid(kx,ky,kz,indexing='ij')
    phase=np.exp(-1j*2*np.pi*((vel[0]*kkx)/nx + (vel[1]*kky)/ny + (vel[2]*kkz)/nz)*t)
    return np.fft.ifftn(U*phase).real
def numflux_LF(a,uL,uR): return 0.5*(a*uL + a*uR) - 0.5*abs(a)*(uR-uL)
def step_advection_3d_with(u, vel, recon, cfl=0.4, t_end=1.0):
    nx,ny,nz=u.shape; ax,ay,az=abs(vel[0]),abs(vel[1]),abs(vel[2]); dt=cfl/max(1e-12,ax+ay+az); t=0.0
    def rhs(curr):
        uL,uR=recon(curr, axis=0); Fx=numflux_LF(vel[0],uL,uR)
        uL,uR=recon(curr, axis=1); Fy=numflux_LF(vel[1],uL,uR)
        uL,uR=recon(curr, axis=2); Fz=numflux_LF(vel[2],uL,uR)
        return -((np.roll(Fx,-1,0)-Fx)+(np.roll(Fy,-1,1)-Fy)+(np.roll(Fz,-1,2)-Fz))
    while t<t_end-1e-12:
        if t+dt>t_end: dt=t_end-t
        u1=u + dt*rhs(u)
        u2=0.75*u + 0.25*(u1 + dt*rhs(u1))
        u  = (1.0/3.0)*u + (2.0/3.0)*(u2 + dt*rhs(u2)); t+=dt
    return u
def make_recon(method):
    if method=='weno_js': return lambda u,axis: weno5_reconstruct(u, axis=axis, mode="JS")
    if method=='weno_z':  return lambda u,axis: weno5_reconstruct(u, axis=axis, mode="Z")
    if method=='eno':     return lambda u,axis: eno_reconstruct(u, axis=axis)
    if method=='tvd_minmod': return lambda u,axis: tvd_minmod_reconstruct(u, axis=axis)
    if method=='lagrange5': return lambda u,axis: (lagrange5_interface(u,axis=axis), lagrange5_interface(u,axis=axis))
    if method=='ppm': return lambda u,axis: ppm_reconstruct(u, axis=axis)
    raise NotImplementedError(method)
