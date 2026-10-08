import numpy as np
from dynamics.graphs import make_graph
from dynamics.system import System
from events.inputs import make_input
from solvers.oracle import solve_oracle
from solvers.synchronous import solve_rk4

def system(n=16,seed=11):
    rng=np.random.default_rng(seed); return System(make_graph('feedback_ring',n,seed),rng.normal(0,.02,(n,n)),rng.normal(0,.02,n),np.ones(n))

def test_oracle_tightening_and_rk4_accuracy():
    s=system(); x,_=make_input('smooth',16,11); t=np.linspace(0,1,101)
    a,_=solve_oracle(s,x,t,rtol=1e-9,atol=1e-11); b,_=solve_oracle(s,x,t,rtol=1e-12,atol=1e-14)
    assert np.max(np.abs(a-b))<1e-7
    z=solve_rk4(s,x,t,dt=.00125)
    assert np.sqrt(np.mean((z-b)**2))<1e-5
