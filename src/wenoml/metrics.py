
import numpy as np
def norms(a,b):
    d=a-b; return {'L1': float(np.mean(np.abs(d))), 'L2': float(np.sqrt(np.mean(d*d))), 'Linf': float(np.max(np.abs(d)))}
def isotropic_spectrum(u):
    nx,ny,nz=u.shape; U=np.fft.fftn(u)
    kx=np.fft.fftfreq(nx)*nx; ky=np.fft.fftfreq(ny)*ny; kz=np.fft.fftfreq(nz)*nz
    kkx,kky,kkz=np.meshgrid(kx,ky,kz,indexing='ij'); k=np.sqrt(kkx**2+kky**2+kkz**2); E=np.abs(U)**2
    kmax=int(k.max()); bins=np.arange(0,kmax+1); E_k=np.zeros_like(bins,dtype=np.float64)
    for i in range(kmax):
        m=(k>=i)&(k<i+1)
        if np.any(m): E_k[i]=E[m].mean()
    return bins, E_k
def spectrum_slope(k,E, kmin=3, kmax=None):
    if kmax is None: kmax=int(0.33*len(k))
    sel=(k>=kmin)&(k<=kmax)&(E>0)
    if np.count_nonzero(sel)<3: return float('nan')
    x=np.log(k[sel]+1e-9); y=np.log(E[sel]); 
    return float(np.polyfit(x,y,1)[0])
