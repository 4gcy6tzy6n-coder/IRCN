import numpy as np

def make_graph(kind: str, n: int, seed: int) -> np.ndarray:
    if kind not in {'chain','star','feedback_ring','modular_sparse'}: raise ValueError(kind)
    W=np.zeros((n,n), dtype=float)
    if kind=='chain':
        W[np.arange(1,n),np.arange(n-1)]=1
    elif kind=='star':
        W[1:,0]=1; W[0,1:]=1
    elif kind=='feedback_ring':
        W[np.arange(n),np.roll(np.arange(n),1)]=1
        W[np.arange(n),np.roll(np.arange(n),-1)]=1
    else:
        rng=np.random.default_rng(seed); block=max(2,int(np.sqrt(n)))
        for i in range(n):
            for j in range(n):
                if i!=j and (i//block==j//block or rng.random()<1/max(n,1)):
                    W[i,j]=1
    rng=np.random.default_rng(seed+100000)
    W *= rng.choice([-1.,1.],size=W.shape)
    rows=np.sum(np.abs(W),axis=1)
    W *= np.minimum(1., .5/np.maximum(rows,1e-15))[:,None]
    return W
