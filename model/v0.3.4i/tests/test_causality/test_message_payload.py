import numpy as np
from dynamics.system import System
from events.scheduler import solve_event

def test_receiver_waits_for_arrived_payload_and_uses_parent_sequence():
    W=np.array([[0.,0.],[.5,0.]])
    U=np.array([[1.],[0.]])
    system=System(W,U,np.zeros(2),np.ones(2))
    x=lambda t:np.array([1.])
    trajectory,meta=solve_event(system,x,np.array([0.,.01,.02,.03]),threshold=.01,max_interval=.05)
    assert trajectory[1,1]==0.0
    assert trajectory[2,1]>0.0
    records=meta['event_log']
    publishes={e['sequence']:e for e in records if e['kind']=='publish'}
    messages=[e for e in records if e['kind']=='message']
    assert messages
    for message in messages:
        parent=publishes[message['parent_sequence']]
        assert parent['time']<=message['time']
        assert parent['node']==message['source']
        assert message['payload']==parent['value']
