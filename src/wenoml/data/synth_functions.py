
import numpy as np
def random_fourier_field(nx, ny, nz, kmin=1, kmax=16, slope=-1.6666667, seed=None):
    rng=np.random.default_rng(seed)
    kx=np.fft.fftfreq(nx)*nx; ky=np.fft.fftfreq(ny)*ny; kz=np.fft.fftfreq(nz)*nz
    kkx,kky,kkz=np.meshgrid(kx,ky,kz,indexing='ij'); k=np.sqrt(kkx**2+kky**2+kkz**2)
    amp=np.zeros_like(k); m=(k>=kmin)&(k<=kmax)&(k>0); amp[m]=1.0/(k[m]**(-slope))
    phase=rng.uniform(0,2*np.pi,size=k.shape); spec=amp*(np.cos(phase)+1j*np.sin(phase)); u=np.fft.ifftn(spec).real
    u=(u-u.min())/(u.max()-u.min()+1e-12); return u
