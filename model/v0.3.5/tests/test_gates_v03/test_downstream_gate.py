import pytest
from ircn.v03.gates import decide

def test_wp2_requires_wp1_pass():
    assert not decide('WP2',{'WP1':'FAIL'}).allowed
    assert decide('WP2',{'WP1':'PASS'}).allowed

def test_unknown_upstream_status_fails_closed():
    result=decide('WP3',{'WP1':'PASS','WP2':'UNKNOWN'})
    assert not result.allowed
    assert 'WP2 status is UNKNOWN' in result.blockers[0]

@pytest.mark.parametrize('wp',['WP4','WP5'])
def test_structural_and_transfer_work_require_wp1_to_wp3(wp):
    status={'WP1':'PASS','WP2':'PASS','WP3':'FAIL','ALGORITHM_FROZEN':'PASS'}
    assert not decide(wp,status).allowed

def test_hardware_requires_fixed_algorithm_and_prior_gates():
    assert not decide('WP6',{'WP1':'PASS','WP2':'PASS','WP3':'PASS'}).allowed
    assert decide('WP6',{'WP1':'PASS','WP2':'PASS','WP3':'PASS','ALGORITHM_FROZEN':'PASS'}).allowed

def test_no_gate_for_unknown_work_package():
    with pytest.raises(ValueError): decide('WP7',{})
