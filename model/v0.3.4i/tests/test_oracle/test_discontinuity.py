import numpy as np
from events.inputs import make_input

def test_piecewise_input_exposes_all_edges():
    x,_=make_input('sparse_pulse',16,11)
    assert x.breakpoints(0,2)==[.02,1.0,1.02]
    assert x.breakpoints(0,0.5)==[.02]
