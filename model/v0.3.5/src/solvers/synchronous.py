import numpy as np

def solve_rk4(system, x, times, dt=0.0025):
    n=system.W.shape[0]; h=np.zeros(n); out=np.empty((len(times),n)); out[0]=h
    t=float(times[0]); kobs=1
    while kobs<len(times):
        target=float(times[kobs]); step=min(dt,target-t)
        if step<=0: out[kobs]=h; kobs+=1; continue
        f=lambda tt,yy: system.rhs(tt,yy,x)
        k1=f(t,h); k2=f(t+step/2,h+step*k1/2); k3=f(t+step/2,h+step*k2/2); k4=f(t+step,h+step*k3)
        h=h+step*(k1+2*k2+2*k3+k4)/6; t+=step
        if t>=target-1e-13: out[kobs]=h; kobs+=1
    return out
