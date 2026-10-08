import numpy as np
from events.scheduler import solve_event
from dynamics.system import System

def test_zero_graph_solution_matches_analytic_decay():
    s=System(np.zeros((1,1)),np.zeros((1,1)),np.array([0.]),np.array([1.]))
    x=lambda t:np.zeros(1)
    out,_=solve_event(s,x,np.array([0.,.1,.2]),threshold=.01)
    np.testing.assert_allclose(out,0.,atol=1e-14)
