import time
from dynamics.graphs import make_graph
from dynamics.system import System
from events.inputs import make_input
from events.scheduler import solve_event
import numpy as np

def test_event_reports_actual_local_evaluations():
    s=System(make_graph("feedback_ring",16,11),np.zeros((16,16)),np.zeros(16),np.ones(16)); x,_=make_input('smooth',16,11); t=np.linspace(0,.1,11)
    _,meta=solve_event(s,x,t)
    assert meta['node_evaluations']>=0
    assert meta['node_evaluations']<len(t)*s.W.shape[0]
