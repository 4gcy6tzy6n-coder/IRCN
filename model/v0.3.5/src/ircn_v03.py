"""IRCN v0.3 candidate: a bounded local-error/event-priority scheduler."""
from events.scheduler import solve_event

def solve_ircn(system,x,times,threshold=.03,gate_dt=.01,max_interval=.05):
    return solve_event(system,x,times,threshold,gate_dt,max_interval)
