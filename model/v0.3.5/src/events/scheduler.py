"""Causal local event engine with held message payloads and lazy exponential prediction."""
import heapq
import itertools
import math
import numpy as np


def solve_event(system, x, times, threshold=.03, gate_dt=.01, max_interval=.05, message_delay=0.0):
    """Simulate with local caches; only arrived messages update a node's inbox.

    State evolves analytically while its local drive is held. State-change threshold
    crossings publish timestamped payloads; the receiver never reads a source's live
    predicted value. A fixed local refresh bounds error from continuous external input.
    Queue order is stable insertion order at equal timestamps.
    """
    if threshold <= 0 or gate_dt <= 0 or max_interval <= 0 or message_delay < 0:
        raise ValueError('threshold, gate_dt, max_interval must be positive and delay nonnegative')
    n=system.W.shape[0]; t0=float(times[0]); t1=float(times[-1])
    anchor=np.zeros(n); anchor_t=np.full(n,t0); target=np.zeros(n); published=np.zeros(n)
    inbox=[{} for _ in range(n)]
    in_edges=[np.flatnonzero(system.W[i]) for i in range(n)]
    out_edges=[np.flatnonzero(system.W[:,i]) for i in range(n)]
    evals=0; queue=[]; seq=itertools.count(); event_log=[]
    local_generation=np.zeros(n,dtype=np.int64); source_version=np.zeros(n,dtype=np.int64)

    def push(at,node,kind,payload=None,parent_sequence=None,source=None,token=None):
        order=next(seq)
        heapq.heappush(queue,(float(at),order,int(node),kind,payload,parent_sequence,source,token))
        return order

    def predict(i,at):
        age=max(0.0,float(at)-anchor_t[i])
        return float(target[i]+(anchor[i]-target[i])*math.exp(-age/system.tau[i]))

    def publish(i,at,parent_sequence):
        value=predict(i,at)
        published[i]=value; source_version[i]+=1
        publish_seq=next(seq)
        event_log.append({'time':float(at),'sequence':publish_seq,'node':int(i),'kind':'publish','value':value,
                          'source_version':int(source_version[i]),'parent_sequence':parent_sequence})
        for dst in out_edges[i]:
            push(at+message_delay,int(dst),'message',payload=value,parent_sequence=publish_seq,source=int(i),token=int(source_version[i]))

    def schedule_local(i,at,parent_sequence):
        local_generation[i]+=1; token=int(local_generation[i])
        y=anchor[i]; pub=published[i]; goal=target[i]; tau=system.tau[i]
        now_delta=abs(y-pub); cross=None
        if now_delta >= threshold*(1-1e-12):
            cross=float(at)
        elif abs(goal-pub)>threshold and abs(y-goal)>1e-15:
            direction=1.0 if goal>pub else -1.0
            boundary=pub+direction*threshold
            ratio=(boundary-goal)/(y-goal)
            if 0.0 < ratio < 1.0: cross=float(at-tau*math.log(ratio))
        fallback=float(at+max_interval)
        next_time=fallback if cross is None else min(cross,fallback)
        if next_time<=t1+1e-14: push(next_time,i,'local',parent_sequence=parent_sequence,token=token)

    def recompute_target(i,at,event_sequence):
        nonlocal evals
        drive=sum(float(system.W[i,j])*inbox[i].get(int(j),0.0) for j in in_edges[i])
        target[i]=float(np.tanh(drive+system.U[i]@x(at)+system.b[i]))
        evals+=1
        schedule_local(i,at,event_sequence)

    # Input updates are external events. Smooth input is sampled on the frozen gate period;
    # discontinuous inputs use their exact, predeclared change times.
    for i in range(n): push(t0,int(i),'input',parent_sequence=None)
    changes=getattr(x,'breakpoints',lambda a,b:[])(t0,t1)
    if getattr(x,'kind',None)=='smooth': changes=list(np.arange(t0+gate_dt,t1+1e-12,gate_dt))
    driven=getattr(x,'driven_ids',range(n))
    for at in changes:
        for i in driven: push(float(at),int(i),'input',parent_sequence=None)

    obs=np.empty((len(times),n)); obs[0]=0.0; oi=1
    while oi<len(times):
        report_time=float(times[oi])
        while queue and queue[0][0]<=report_time+1e-14:
            at,event_sequence,node,kind,payload,parent_sequence,source,token=heapq.heappop(queue)
            if kind=='local' and token!=local_generation[node]: continue
            event={'time':at,'sequence':event_sequence,'node':node,'kind':kind,'parent_sequence':parent_sequence}
            if kind=='message': event.update(source=source,source_version=token,payload=payload)
            event_log.append(event)
            anchor[node]=predict(node,at); anchor_t[node]=at
            if kind=='message': inbox[node][source]=float(payload)
            elif kind=='local' and abs(anchor[node]-published[node])>=threshold*(1-1e-10):
                publish(node,at,event_sequence)
            recompute_target(node,at,event_sequence)
        obs[oi]=[predict(i,report_time) for i in range(n)]
        oi+=1
    return obs,{'node_evaluations':evals,'queue_pushes':next(seq),'queue_remaining':len(queue),
                'output_materializations':len(times)*n,'event_log':event_log}
