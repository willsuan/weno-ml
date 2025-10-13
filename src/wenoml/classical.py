
import numpy as np
def _roll(x,s,ax): return np.roll(x,s,axis=ax)
def eno_reconstruct(u, axis=-1):
    um2=_roll(u,+2,axis); um1=_roll(u,+1,axis); u0=u; up1=_roll(u,-1,axis); up2=_roll(u,-2,axis)
    q0=(2*um2 - 7*um1 + 11*u0)/6; q1=(-um1 + 5*u0 + 2*up1)/6; q2=(2*u0 + 5*up1 - up2)/6
    b0=(13/12)*(um2-2*um1+u0)**2 + 0.25*(um2 - 4*um1 + 3*u0)**2
    b1=(13/12)*(um1-2*u0+up1)**2 + 0.25*(um1 - up1)**2
    b2=(13/12)*(u0-2*up1+up2)**2 + 0.25*(3*u0 - 4*up1 + up2)**2
    k0=(b0<=b1)&(b0<=b2); k1=(b1<b0)&(b1<=b2); k2=~(k0|k1)
    uL=np.where(k0,q0,np.where(k1,q1,q2))
    xr=_roll(u,-1,axis); um2=_roll(xr,-2,axis); um1=_roll(xr,-1,axis); u0=xr; up1=_roll(xr,1,axis); up2=_roll(xr,2,axis)
    q0=(2*um2 - 7*um1 + 11*u0)/6; q1=(-um1 + 5*u0 + 2*up1)/6; q2=(2*u0 + 5*up1 - up2)/6
    b0=(13/12)*(um2-2*um1+u0)**2 + 0.25*(um2 - 4*um1 + 3*u0)**2
    b1=(13/12)*(um1-2*u0+up1)**2 + 0.25*(um1 - up1)**2
    b2=(13/12)*(u0-2*up1+up2)**2 + 0.25*(3*u0 - 4*up1 + up2)**2
    k0=(b0<=b1)&(b0<=b2); k1=(b1<b0)&(b1<=b2); k2=~(k0|k1)
    uR=np.where(k0,q0,np.where(k1,q1,q2))
    return uL,uR
def minmod(a,b): return 0.5*(np.sign(a)+np.sign(b))*np.minimum(np.abs(a), np.abs(b))
def tvd_minmod_reconstruct(u, axis=-1):
    df=_roll(u,-1,axis)-u; db=u-_roll(u,+1,axis); s=minmod(df,db)
    uL=u+0.5*s; s_r=_roll(s,-1,axis); uR=_roll(u,-1,axis)-0.5*s_r; return uL,uR
def lagrange5_interface(u, axis=-1):
    um2=_roll(u,+2,axis); um1=_roll(u,+1,axis); u0=u; up1=_roll(u,-1,axis); up2=_roll(u,-2,axis); up3=_roll(u,-3,axis)
    return (3*um2 - 25*um1 + 150*u0 + 150*up1 - 25*up2 + 3*up3)/256.0
def ppm_reconstruct(u, axis=-1):
    um1=_roll(u,+1,axis); up1=_roll(u,-1,axis)
    def vanleer(a,b):
        s = np.sign(a)+np.sign(b); return np.where(s!=0, (2*a*b)/(a+b+1e-12), 0.0)
    slope = vanleer(u-um1, up1-u)
    uL = u - 0.5*slope; uR = u + 0.5*slope
    umin = np.minimum.reduce([um1,u,up1]); umax=np.maximum.reduce([um1,u,up1])
    uL = np.clip(uL, umin, umax); uR = np.clip(uR, umin, umax)
    return uR, _roll(uL,-1,axis)
