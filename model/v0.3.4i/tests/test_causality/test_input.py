import numpy as np
from events.inputs import make_input

def test_input_is_deterministic_and_nonanticipating():
    a,_=make_input('sparse_pulse',32,23); b,_=make_input('sparse_pulse',32,23)
    for t in [0,.01,.019,.02,.5]: np.testing.assert_array_equal(a(t),b(t))
    assert np.count_nonzero(a(.01))>0
    assert np.count_nonzero(a(.03))==0
