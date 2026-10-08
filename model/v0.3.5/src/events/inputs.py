import numpy as np

def _pulse_is_on(t, period, width):
    """Use a boundary tolerance so generated events agree at float breakpoints."""
    t=float(t); phase=t % period
    tol=8*np.finfo(float).eps*max(1.0,abs(t),abs(period))
    return phase < width-tol

def make_input(kind,n,seed,driven_fraction=.05):
    rng=np.random.default_rng(seed); count=max(1,int(np.ceil(n*driven_fraction))); ids=np.sort(rng.choice(n,count,replace=False))
    def x(t):
        z=np.zeros(n)
        if kind=='smooth': z[ids]=np.sin(2*np.pi*.5*t)
        elif kind=='sparse_pulse': z[ids]=1.0 if _pulse_is_on(t,1.0,.02) else 0.0
        elif kind=='dense_burst': z[ids]=1.0 if _pulse_is_on(t,.125,.02) else 0.0
        elif kind=='state_switch': z[ids]=1.0 if int(t/.25)%2==0 else -1.0
        else: raise ValueError(kind)
        return z
    x.driven_ids=ids
    x.kind=kind
    def breakpoints(t0,t1):
        if kind=='smooth': return []
        period={'sparse_pulse':1.,'dense_burst':.125,'state_switch':.25}[kind]
        changes=[.02] if kind!='state_switch' else [0.]
        out=[]
        k0=max(0,int(np.floor(t0/period))-1); k1=int(np.ceil(t1/period))+1
        for k in range(k0,k1+1):
            for phase in changes:
                t=k*period+phase
                if t0<t<t1: out.append(t)
            # periodic pulse falling edge
            if kind in ('sparse_pulse','dense_burst'):
                t=k*period
                if t0<t<t1: out.append(t)
        return sorted(set(out))
    x.breakpoints=breakpoints
    return x, ids
