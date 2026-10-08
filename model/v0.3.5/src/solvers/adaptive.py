from scipy.integrate import solve_ivp
import numpy as np

def solve_adaptive(system,x,times,rtol=1e-6,atol=1e-9):
    out=solve_ivp(lambda t,y:system.rhs(t,y,x),(times[0],times[-1]),np.zeros(system.W.shape[0]),
                  method='DOP853',rtol=rtol,atol=atol,t_eval=times)
    if not out.success: raise RuntimeError(out.message)
    return out.y.T, {'nfev':out.nfev}
