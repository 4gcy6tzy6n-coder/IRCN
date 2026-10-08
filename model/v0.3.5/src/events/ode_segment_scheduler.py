"""Exploratory local ODE solver over causally received neighbor trajectory segments."""
import heapq
import itertools
import numpy as np
from scipy.integrate import solve_ivp

class Segment:
    def __init__(self,start,end,solution,version,component=0):
        self.start=float(start);self.end=float(end);self.solution=solution;self.version=int(version);self.component=int(component)
    def value(self,t):return float(self.solution(float(t))[self.component])


def solve_ode_segments(system,x,times,segment_tolerance=1e-3,gate_dt=.01,max_interval=.05,input_mode='linear_gate',delivery_mode='edge_messages',local_rtol=1e-9,local_atol=1e-11,local_batch=False,local_batch_size=None):
    """Calibration-only scalar integrations from causally received source segments."""
    if segment_tolerance<=0 or gate_dt<=0 or max_interval<=0:raise ValueError('positive tolerances and intervals required')
    if input_mode not in {'linear_gate','exact_exogenous'}:raise ValueError('input_mode must be linear_gate or exact_exogenous')
    if delivery_mode not in {'edge_messages','broadcast_batch'}:raise ValueError('delivery_mode must be edge_messages or broadcast_batch')
    if local_batch_size is not None and (not isinstance(local_batch_size,(int,np.integer)) or local_batch_size<1):raise ValueError('local_batch_size must be a positive integer')
    n=system.W.shape[0];t0=float(times[0]);t1=float(times[-1])
    in_edges=[np.flatnonzero(system.W[i]) for i in range(n)];out_edges=[np.flatnonzero(system.W[:,i]) for i in range(n)]
    own=[None for _ in range(n)];published=[None for _ in range(n)];inbox=[{} for _ in range(n)]
    generation=np.zeros(n,dtype=np.int64);version=np.zeros(n,dtype=np.int64);queue=[];seq=itertools.count();event_log=[];rhs_evaluations=0;heap_pushes=0;message_deliveries=0
    x_anchor=np.asarray(x(t0),dtype=float);x_time=t0;is_smooth=getattr(x,'kind',None)=='smooth'
    x_slope=(np.asarray(x(t0),dtype=float)-np.asarray(x(t0-gate_dt),dtype=float))/gate_dt if is_smooth else np.zeros_like(x_anchor)
    affected=np.flatnonzero(np.any(np.abs(system.U)>0,axis=1))
    input_breaks=sorted(float(q) for q in getattr(x,'breakpoints',lambda a,b:[])(t0,t1))
    def push(at,node,kind,payload=None,parent=None,source=None,token=None):
        nonlocal heap_pushes
        order=next(seq);heapq.heappush(queue,(float(at),order,int(node),kind,payload,parent,source,token));heap_pushes+=1;return order
    def state_at(i,t):
        seg=own[i]
        if seg is None:return 0.0
        return seg.value(min(max(float(t),seg.start),seg.end))
    def input_at(t):
        if input_mode=='exact_exogenous':return np.asarray(x(float(t)),dtype=float)
        return x_anchor+x_slope*(float(t)-x_time)
    def diff_segments(old,new,t):
        if old is None:return float('inf')
        end=min(old.end,new.end)
        if end<=t+1e-12:return 0.0
        return max(abs(old.value(q)-new.value(q)) for q in np.linspace(t,end,9))
    def segment_end(i,t):
        horizons=[t+max_interval,t1]
        horizons.extend(seg.end for seg in inbox[i].values() if seg.end>t+1e-12)
        if input_mode=='exact_exogenous' and np.any(system.U[i]):
            next_break=next((q for q in input_breaks if q>t+1e-12),None)
            if next_break is not None:horizons.append(next_break)
        return min(horizons)
    def integrate_nodes(nodes,t):
        nonlocal rhs_evaluations
        end=segment_end(nodes[0],t)
        y0=np.asarray([state_at(i,t) for i in nodes])
        if end<=t+1e-12:return [Segment(t,t,lambda q,y=y0: y,version[i],k) for k,i in enumerate(nodes)]
        def rhs(q,y):
            drive=np.empty(len(nodes),dtype=float)
            ext=input_at(q)
            for k,i in enumerate(nodes):
                value=float(system.b[i]+system.U[i]@ext)
                for j in in_edges[i]:
                    seg=inbox[i].get(int(j))
                    if seg is not None:value+=float(system.W[i,j])*seg.value(q)
                drive[k]=value
            return (-y+np.tanh(drive))/system.tau[nodes]
        # DOP853 controls a vector RMS norm. Tighten by sqrt(batch size) so a
        # component cannot receive a looser normalized local error allowance
        # merely because unrelated nodes share the integrator instance.
        error_scale=np.sqrt(len(nodes))
        result=solve_ivp(rhs,(t,end),y0,method='DOP853',rtol=local_rtol/error_scale,atol=local_atol/error_scale,dense_output=True)
        rhs_evaluations+=result.nfev
        if not result.success:raise RuntimeError(result.message)
        return [Segment(t,end,result.sol,version[i],k) for k,i in enumerate(nodes)]
    def publish(i,seg,t,parent):
        version[i]+=1;seg.version=int(version[i]);published[i]=seg;pubseq=next(seq)
        event_log.append({'time':t,'sequence':pubseq,'node':i,'kind':'segment_publish','version':int(version[i]),'parent_sequence':parent,'valid_until':seg.end})
        if delivery_mode=='edge_messages':
            for dst in out_edges[i]:push(t,int(dst),'message',seg,pubseq,i,int(version[i]))
        else:
            push(t,int(i),'broadcast',seg,pubseq,int(i),int(version[i]))
    def rebuild_many(nodes,t,parents):
        nodes=sorted(nodes)
        tokens={}
        oldpubs={}
        for i in nodes:
            generation[i]+=1;tokens[i]=int(generation[i]);oldpubs[i]=published[i]
        # Group only causally touched nodes with identical segment horizons.
        groups={}
        for i in nodes:groups.setdefault(segment_end(i,t),[]).append(i)
        segments={}
        for _,group in sorted(groups.items()):
            batch_size=local_batch_size or len(group)
            if local_batch and len(group)>1:
                for start in range(0,len(group),batch_size):
                    chunk=group[start:start+batch_size]
                    solved=integrate_nodes(chunk,t)
                    segments.update(zip(chunk,solved))
            else:
                for i in group:segments[i]=integrate_nodes([i],t)[0]
        for i in nodes:
            seg=segments[i];own[i]=seg;oldpub=oldpubs[i]
            # A segment ending at this time has no valid future payload to hold.
            expired=oldpub is not None and oldpub.end<=t+1e-12
            has_future=seg.end>t+1e-12
            if has_future and (oldpub is None or expired or diff_segments(oldpub,seg,t)>=segment_tolerance):publish(i,seg,t,parents.get(i))
            if has_future:push(seg.end,i,'refresh',parent=None,token=tokens[i])
    # External inputs are delivered only at their event times; smooth input is linearly
    # extrapolated from the present and its previous sample, never queried in the future.
    push(t0,-1,'input')
    # Every node has autonomous bias dynamics even when its external input row is zero.
    # Initialize one local segment per node so those states can evolve and propagate.
    for i in range(n):push(t0,int(i),'initialize')
    changes=input_breaks
    if input_mode=='linear_gate' and is_smooth:changes=list(np.arange(t0+gate_dt,t1+1e-12,gate_dt))
    if input_mode=='exact_exogenous' and is_smooth:changes=[]
    for t in changes:push(t,-1,'input')
    obs=np.zeros((len(times),n));oi=1
    while oi<len(times):
        report=float(times[oi]);same_time_events=0
        while queue and queue[0][0]<=report+1e-14:
            at=queue[0][0];batch=[]
            while queue and queue[0][0]<=at+1e-14:batch.append(heapq.heappop(queue))
            touched=set();parents={}
            for et,order,node,kind,payload,parent,source,token in batch:
                if kind=='refresh' and token!=generation[node]:continue
                event={'time':et,'sequence':order,'node':node,'kind':kind,'parent_sequence':parent}
                if kind=='message':
                    inbox[node][source]=payload;event.update(source=source,source_version=token,segment_end=payload.end)
                    touched.add(node);parents[node]=order
                    message_deliveries+=1
                elif kind=='broadcast':
                    for dst in out_edges[node]:
                        delivery_seq=next(seq)
                        inbox[int(dst)][node]=payload
                        event_log.append({'time':et,'sequence':delivery_seq,'node':int(dst),'kind':'message','parent_sequence':parent,'source':node,'source_version':token,'segment_end':payload.end})
                        touched.add(int(dst));parents[int(dst)]=delivery_seq
                        message_deliveries+=1
                elif kind=='input':
                    old=x_anchor.copy();new=np.asarray(x(et),dtype=float)
                    if is_smooth and input_mode=='linear_gate':x_slope=(new-np.asarray(x(et-gate_dt),dtype=float))/gate_dt
                    else:x_slope=np.zeros_like(new)
                    x_anchor=new;x_time=float(et);touched.update(int(i) for i in affected)
                    parents.update({int(i):order for i in affected});event['changed_dimensions']=int(np.count_nonzero(new!=old))
                elif kind=='initialize':touched.add(node);parents[node]=order
                elif kind=='refresh':touched.add(node);parents[node]=order
                event_log.append(event)
            if touched:rebuild_many(touched,at,parents)
            same_time_events+=len(batch)
            if same_time_events>20000:raise RuntimeError(f'same-time event cascade limit exceeded at t={at}')
        obs[oi]=[state_at(i,report) for i in range(n)];oi+=1
    return obs,{'event_log':event_log,'event_count':len(event_log),'rhs_evaluations':rhs_evaluations,
                'queue_pushes':heap_pushes,'message_deliveries':message_deliveries,
                'output_materializations':len(times)*n,'delivery_mode':delivery_mode}
