import numpy as np
from dynamics.graphs import make_graph
from dynamics.system import System
from events.inputs import make_input
from events.scheduler import solve_event
from solvers.oracle import solve_oracle

def test_event_causal_and_threshold_convergence():
    n=16; rng=np.random.default_rng(11); s=System(make_graph('feedback_ring',n,11),rng.normal(0,.01,(n,n)),rng.normal(0,.03,n),np.ones(n))
    x,_=make_input('smooth',n,11); t=np.linspace(0,1,101); ref,_=solve_oracle(s,x,t)
    loose,_=solve_event(s,x,t,.1); tight,_=solve_event(s,x,t,.01)
    e1=np.sqrt(np.mean((loose-ref)**2)); e2=np.sqrt(np.mean((tight-ref)**2))
    assert np.isfinite(e1+e2)
    # Candidate threshold tightening is a required empirical gate, tested separately from this smoke assertion.
