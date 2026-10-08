import numpy as np
from scipy.integrate import solve_ivp

def solve_oracle(system, x, times, rtol=1e-11, atol=1e-13):
    """Independent DOP853 integration, segmented at known input discontinuities."""
    y=np.zeros(system.W.shape[0]); out=np.empty((len(times),len(y))); nfev=0
    breaks=getattr(x,'breakpoints',lambda a,b:[])(float(times[0]),float(times[-1]))
    cuts=sorted(set([float(times[0]),*breaks,float(times[-1])]))
    for left,right in zip(cuts[:-1],cuts[1:]):
        sol=solve_ivp(lambda t,h:system.rhs(t,h,x),(left,right),y,method='DOP853',rtol=rtol,atol=atol,dense_output=True)
        if not sol.success: raise RuntimeError(sol.message)
        nfev+=sol.nfev
        mask=(times>=left-1e-14)&(times<=right+1e-14)
        out[mask]=sol.sol(times[mask]).T
        y=sol.y[:,-1]
    class Info: pass
    info=Info(); info.nfev=nfev
    return out,info
