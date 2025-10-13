
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Local imports
from wenoml.data.synth_functions import random_fourier_field
from wenoml.eval.rollout_bench import numflux_LF
from wenoml.weno5 import weno5_reconstruct
from wenoml.pde.shallow_water2d import reconstruct as sw_reconstruct, rusanov_flux as sw_flux

def advection3d_frames(nx=64, ny=64, nz=64, t_end=0.5, vel=(1.0,-0.5,0.25), nframes=40, seed=2025):
    u = random_fourier_field(nx,ny,nz, kmin=1, kmax=18, slope=-1.6666667, seed=seed)
    dt = t_end / nframes
    frames = []
    def rhs(curr):
        uL,uR=weno5_reconstruct(curr, axis=0, mode="Z"); Fx=numflux_LF(vel[0],uL,uR)
        uL,uR=weno5_reconstruct(curr, axis=1, mode="Z"); Fy=numflux_LF(vel[1],uL,uR)
        uL,uR=weno5_reconstruct(curr, axis=2, mode="Z"); Fz=numflux_LF(vel[2],uL,uR)
        return -((np.roll(Fx,-1,0)-Fx)+(np.roll(Fy,-1,1)-Fy)+(np.roll(Fz,-1,2)-Fz))
    for k in range(nframes):
        frames.append(u[nx//2].copy())
        u1 = u + dt*rhs(u)
        u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1))
        u  = (1.0/3.0)*u + (2.0/3.0)*(u2 + dt*rhs(u2))
    return frames

def shallow_water_frames(nx=192, ny=128, t_end=0.5, nframes=60, g=9.81, fcor=1e-4):
    h=np.ones((nx,ny), np.float64); h[:nx//2]+=0.2; hu=np.zeros_like(h); hv=np.zeros_like(h)
    U=np.stack([h,hu,hv], -1)
    frames=[]; dt = t_end / nframes
    def rhs(curr):
        ULx,URx = sw_reconstruct(curr, method='weno_z', axis=0); Fx=sw_flux(ULx,URx,axis=0,g=g,fcor=fcor)
        ULy,URy = sw_reconstruct(curr, method='weno_z', axis=1); Fy=sw_flux(ULy,URy,axis=1,g=g,fcor=fcor)
        div=(np.roll(Fx,-1,0)-Fx)+(np.roll(Fy,-1,1)-Fy)
        h,hu,hv = curr[...,0],curr[...,1],curr[...,2]
        S = np.stack([np.zeros_like(h), fcor*(-hv), fcor*(hu)], -1)
        return -(div - S)
    for k in range(nframes):
        frames.append(U[...,0].T.copy())
        U1=U + dt*rhs(U)
        U2=0.75*U + 0.25*(U1 + dt*rhs(U1))
        U = (1.0/3.0)*U + (2.0/3.0)*(U2 + dt*rhs(U2))
    return frames

def write_gif(frames, out_path, interval_ms=60):
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], origin='lower', interpolation='nearest')
    ax.set_title(os.path.basename(out_path))
    def update(i):
        im.set_data(frames[i])
        return (im,)
    anim = FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=max(1,int(1000/interval_ms))))
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--case', type=str, required=True, choices=['advection3d_slice','shallow_water_2d'])
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--frames', type=int, default=60)
    ap.add_argument('--duration', type=float, default=0.5)
    args=ap.parse_args()
    if args.case=='advection3d_slice':
        frames = advection3d_frames(t_end=args.duration, nframes=args.frames)
        write_gif(frames, args.out, interval_ms=int(1000*args.duration/max(1,args.frames)))
    elif args.case=='shallow_water_2d':
        frames = shallow_water_frames(t_end=args.duration, nframes=args.frames)
        write_gif(frames, args.out, interval_ms=int(1000*args.duration/max(1,args.frames)))

if __name__=='__main__':
    main()
