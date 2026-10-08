"""Frozen task-to-system mapping for prospective IRCN v0.3.1 calibration."""
import numpy as np
from dynamics.graphs import make_graph
from dynamics.system import System


def make_task_system(graph_kind: str, n: int, seed: int, driven_ids) -> System:
    """Construct dynamics where only designated nodes receive external input.

    Independent deterministic RNG streams keep graph, bias/time constants, and input
    gains reproducible and prevent adding one random tensor from shifting other params.
    """
    ids=np.asarray(driven_ids,dtype=int)
    if ids.ndim!=1 or np.any(ids<0) or np.any(ids>=n) or len(np.unique(ids))!=len(ids):
        raise ValueError('driven_ids must be unique valid node indices')
    W=make_graph(graph_kind,n,seed)
    rng_state=np.random.default_rng(seed+5000+n)
    b=rng_state.normal(0,.05,n)
    tau=np.exp(rng_state.uniform(np.log(.5),np.log(2.0),n))
    rng_input=np.random.default_rng(seed+300000+n)
    U=np.zeros((n,n),dtype=float)
    U[ids,ids]=rng_input.normal(0,.05,len(ids))
    return System(W,U,b,tau)
