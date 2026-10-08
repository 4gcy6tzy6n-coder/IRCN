import numpy as np
from dynamics.tasks import make_task_system

def test_only_selected_nodes_receive_external_input():
    n=32;ids=np.array([2,7])
    system=make_task_system('feedback_ring',n,101,ids)
    non_driven=np.setdiff1d(np.arange(n),ids)
    assert np.count_nonzero(system.U[non_driven])==0
    assert np.count_nonzero(system.U[ids])==len(ids)
    x=np.zeros(n);x[ids]=1.0
    external=system.U@x
    assert np.flatnonzero(external).tolist()==ids.tolist()

def test_task_system_parameters_are_deterministic_and_stream_separated():
    a=make_task_system('modular_sparse',32,131,[1,4])
    b=make_task_system('modular_sparse',32,131,[1,4])
    np.testing.assert_array_equal(a.W,b.W)
    np.testing.assert_array_equal(a.U,b.U)
    np.testing.assert_array_equal(a.b,b.b)
    np.testing.assert_array_equal(a.tau,b.tau)
