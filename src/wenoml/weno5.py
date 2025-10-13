
import numpy as np
def _roll(x,s,ax): return np.roll(x,s,axis=ax)
def weno5_weights(u, axis=-1, eps=1e-6, p=2.0, mode="JS"):
    um2=_roll(u,+2,axis); um1=_roll(u,+1,axis); u0=u; up1=_roll(u,-1,axis); up2=_roll(u,-2,axis)
    b0=(13/12)*(um2-2*um1+u0)**2 + 0.25*(um2-4*um1+3*u0)**2
    b1=(13/12)*(um1-2*u0+up1)**2 + 0.25*(um1-up1)**2
    b2=(13/12)*(u0-2*up1+up2)**2 + 0.25*(3*u0-4*up1+up2)**2
    d0,d1,d2=0.1,0.6,0.3
    if mode=="JS":
        a0=d0/((np.abs(b0)+eps)**p); a1=d1/((np.abs(b1)+eps)**p); a2=d2/((np.abs(b2)+eps)**p)
    else:
        tau=np.abs(b2-b0); a0=d0*(1+(tau/(np.abs(b0)+eps))**p); a1=d1*(1+(tau/(np.abs(b1)+eps))**p); a2=d2*(1+(tau/(np.abs(b2)+eps))**p)
    A=np.stack([a0,a1,a2],-1); return A/np.sum(A,-1,keepdims=True)
def weno5_reconstruct(u, axis=-1, eps=1e-6, p=2.0, mode="JS"):
    um2=_roll(u,+2,axis); um1=_roll(u,+1,axis); u0=u; up1=_roll(u,-1,axis); up2=_roll(u,-2,axis)
    q0=(2*um2 - 7*um1 + 11*u0)/6; q1=(-um1 + 5*u0 + 2*up1)/6; q2=(2*u0 + 5*up1 - up2)/6
    w=weno5_weights(u, axis=axis, eps=eps, p=p, mode=mode); uL=(w[...,0]*q0 + w[...,1]*q1 + w[...,2]*q2)
    xr=_roll(u,-1,axis); um2=_roll(xr,-2,axis); um1=_roll(xr,-1,axis); u0=xr; up1=_roll(xr,1,axis); up2=_roll(xr,2,axis)
    q0=(2*um2 - 7*um1 + 11*u0)/6; q1=(-um1 + 5*u0 + 2*up1)/6; q2=(2*u0 + 5*up1 - up2)/6
    w=weno5_weights(xr, axis=axis, eps=eps, p=p, mode=mode); uR=(w[...,0]*q0 + w[...,1]*q1 + w[...,2]*q2)
    return uL,uR
