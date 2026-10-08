"""Generate WP2 paired local-skip counterfactual labels under the frozen contract."""
from __future__ import annotations
import csv, hashlib, json, math, time, traceback, platform, sys, scipy, numpy, datetime
from pathlib import Path
import numpy as np
import yaml
from scipy.integrate import solve_ivp
from dynamics.tasks import make_task_system
from events.inputs import make_input
from events.ode_segment_scheduler import solve_ode_segments
from solvers.oracle import solve_oracle

ROOT=Path(__file__).resolve().parents[2]
CONTRACT=ROOT/'contracts/V03_WP2_INFLUENCE_v1.yaml'
OUT=ROOT/'reports/WP2/run_v1'
C=yaml.safe_load(CONTRACT.read_text()); CONTRACT_SHA=hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
TIMES=np.linspace(0.,2.,201); ANCHORS=C['sampling']['snapshot_times_seconds']; DT=C['intervention']['delay_seconds']; H=C['intervention']['output_horizon_seconds']; SAMPLE=C['intervention']['output_sampling_seconds']
SCHED=C['system']['candidate_scheduler_sha256']

def hash_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def selected_nodes(graph,n,task,seed,t,split):
 if split!='train': return np.arange(n,dtype=int)
 key=f"{C['contract_id']}|{n}|{graph}|{task}|{seed}|{t:.8f}".encode()
 rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8],'big'))
 return np.sort(rng.choice(n,size=min(C['sampling']['training_nodes_per_snapshot'],n),replace=False))

def linear_input(x,t,dt):
 now=np.asarray(x(float(t)),dtype=float); prev=np.asarray(x(float(t-dt)),dtype=float); slope=(now-prev)/dt
 return lambda q: now+(float(q)-t)*slope

def features_at(system,x,t,h):
 """Features only from current state/parameters and current/past input."""
 w=system.W; absw=np.abs(w); x_now=np.asarray(x(float(t)),dtype=float); xpast=np.asarray(x(float(t-DT)),dtype=float)
 xlin=linear_input(x,t,DT)
 f0=system.rhs(t,h,xlin)
 hmid=h+0.5*DT*f0
 fmid=system.rhs(t+0.5*DT,hmid,xlin)
 trunc=np.abs(0.5*DT*(fmid-f0))
 return np.column_stack((np.abs(h),np.abs(f0),trunc,absw.sum(axis=1),absw.sum(axis=0),absw@np.abs(h),system.tau,np.abs(system.U@x_now))), {'input_now':x_now,'input_past':xpast}

def integrate_window(system,x,h0,t0,*,clamp_node=None):
 ts=np.round(np.arange(t0,t0+H+SAMPLE/2,SAMPLE),12); end=float(ts[-1]); release=t0+DT
 breaks=list(getattr(x,'breakpoints',lambda a,b:[])(t0,end))
 cuts=sorted(set([float(t0),end,*[float(q) for q in breaks if t0<q<end],*([release] if clamp_node is not None and t0<release<end else [])]))
 out=np.empty((len(ts),len(h0))); nfev=0; y=h0.copy()
 for left,right in zip(cuts[:-1],cuts[1:]):
  hold=clamp_node is not None and left<release-1e-13
  is_input_edge=any(abs(right-q)<1e-12 for q in breaks)
  def rhs(q,z):
   qx=np.nextafter(right,left) if is_input_edge and q>=right else q
   dz=system.rhs(float(qx),z,x)
   if hold: dz[int(clamp_node)]=0.0
   return dz
  sol=solve_ivp(rhs,(left,right),y,method='DOP853',rtol=float(C['intervention']['oracle_rtol']),atol=float(C['intervention']['oracle_atol']),dense_output=True)
  if not sol.success: raise RuntimeError(sol.message)
  nfev+=sol.nfev
  mask=(ts>=left-1e-13)&(ts<=right+1e-13)
  out[mask]=sol.sol(ts[mask]).T
  y=sol.y[:,-1]
 return out,nfev

def local_node_update_seconds(system,x,h,t,node):
 """Time one scalar local DOP853 update with neighbor states held at snapshot."""
 def rhs(q,z):
  ht=h.copy(); ht[node]=z[0]
  return [system.rhs_node(node,ht,np.asarray(x(float(q)),dtype=float))]
 start=time.perf_counter(); sol=solve_ivp(rhs,(0.,DT),[float(h[node])],method='DOP853',rtol=1e-9,atol=1e-11); elapsed=time.perf_counter()-start
 if not sol.success:raise RuntimeError(sol.message)
 return elapsed

def row_for(graph,n,task,seed,split,on_record):
 instance_start=time.perf_counter(); x,driven=make_input(task,n,seed); system=make_task_system(graph,n,seed,driven)
 trace=np.asarray([x(float(t)) for t in TIMES],dtype='<f8'); input_hash=hashlib.sha256(trace.tobytes()).hexdigest()
 start=time.perf_counter()
 candidate,meta=solve_ode_segments(system,x,TIMES,segment_tolerance=1e-7,max_interval=.02,input_mode='exact_exogenous',delivery_mode='broadcast_batch',local_rtol=1e-9,local_atol=1e-11,local_batch=True,local_batch_size=2)
 scheduler_seconds=time.perf_counter()-start
 oracle,oracle_info=solve_oracle(system,x,TIMES,rtol=1e-13,atol=1e-15)
 scale=max(float(np.sqrt(np.mean(oracle*oracle))),1e-12)
 candidate_nrmse=float(np.sqrt(np.mean((candidate-oracle)**2))/scale)
 candidate_max_abs=float(np.max(np.abs(candidate-oracle)))
 records=[]
 for t in ANCHORS:
  idx=int(round(t/.01)); horacle=oracle[idx]
  hfeat=candidate[idx].copy()
  for _ in range(5): features_at(system,x,t,candidate[idx].copy())
  fstart=time.perf_counter()
  for _ in range(30):
   hfeat=candidate[idx].copy(); all_features,aux=features_at(system,x,t,hfeat)
  feature_extract_seconds=(time.perf_counter()-fstart)/30
  nodes=selected_nodes(graph,n,task,seed,t,split)
  baseline,base_nfev=integrate_window(system,x,horacle,t)
  norm=max(float(np.sqrt(np.mean(baseline*baseline))),1e-12)
  per_node_update=[]
  for node in nodes:
   skipped,skip_nfev=integrate_window(system,x,horacle,t,clamp_node=int(node))
   influence=float(np.sqrt(np.mean((skipped-baseline)**2))/norm)
   node_cost=local_node_update_seconds(system,x,hfeat,t,int(node))
   per_node_update.append(node_cost)
   record={'contract_sha256':CONTRACT_SHA,'graph':graph,'n':n,'task':task,'seed':seed,'snapshot_time':t,'node':int(node),'input_sha256':input_hash,'influence':influence,'per_time_difference_rms':np.sqrt(np.mean((skipped-baseline)**2,axis=1)).tolist(),'per_time_baseline_rms':np.sqrt(np.mean(baseline**2,axis=1)).tolist(),'candidate_snapshot_abs_error':float(abs(hfeat[node]-horacle[node])),'feature_extract_seconds':feature_extract_seconds,'local_update_seconds':node_cost,'baseline_nfev':base_nfev,'skip_nfev':skip_nfev,'features':all_features[node].tolist(),'split':split}
   records.append(record);on_record(record)
 return records,{'graph':graph,'n':n,'task':task,'seed':seed,'input_sha256':input_hash,'scheduler_seconds':scheduler_seconds,'scheduler_meta':{k:v for k,v in meta.items() if k!='event_log'},'oracle_nfev':int(oracle_info.nfev),'label_generation_wall_seconds':time.perf_counter()-instance_start,'candidate_full_trace_nrmse':candidate_nrmse,'candidate_full_trace_max_abs':candidate_max_abs,'snapshot_node_count':sum(len(selected_nodes(graph,n,task,seed,t,split)) for t in ANCHORS)}

def case_worker(args):
 split,task,graph,n,seed=args
 emitted=[]
 try:
  records,meta=row_for(graph,n,task,seed,split,emitted.append);meta.update(split=split)
  meta['code_sha256']={p:hash_file(ROOT/p) for p in ['src/events/ode_segment_scheduler.py','src/events/inputs.py','src/dynamics/tasks.py','src/solvers/oracle.py','experiments/WP1_v034i/calibrate_local_batch2.py','experiments/WP2/generate_labels.py']}
  return emitted,meta,None
 except Exception as exc:
  failure={'split':split,'graph':graph,'n':n,'task':task,'seed':seed,'error':repr(exc),'traceback':traceback.format_exc()}
  return emitted,None,failure

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 seed_keys={'train':'training_seeds_by_n','validation':'validation_seeds_by_n','test':'test_seeds_by_n'}
 seeds={sp:{int(n):vals for n,vals in C['split'][key].items()} for sp,key in seed_keys.items()}
 tasks={'train':C['split']['train_validation_tasks'],'validation':C['split']['train_validation_tasks'],'test':C['split']['test_tasks']}
 raw=OUT/'labels_raw.jsonl'; instances=OUT/'instances.jsonl'; failures_file=OUT/'failures.jsonl'
 raw.touch(exist_ok=True);instances.touch(exist_ok=True);failures_file.touch(exist_ok=True)
 prior_meta=[json.loads(line) for line in instances.read_text().splitlines() if line.strip()]
 done={(z['split'],z['graph'],int(z['n']),z['task'],int(z['seed'])) for z in prior_meta}
 # Preserve but exclude partial rows for any instance lacking a completion record.
 if raw.stat().st_size:
  old_rows=[json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
  incomplete=[r for r in old_rows if (r['split'],r['graph'],int(r['n']),r['task'],int(r['seed'])) not in done]
  if incomplete:
   history=OUT/'partial_rows_history.jsonl'
   with history.open('a') as f:
    for r in incomplete:f.write(json.dumps({'prior_partial_row':r},sort_keys=True)+'\n')
   complete_rows=[r for r in old_rows if (r['split'],r['graph'],int(r['n']),r['task'],int(r['seed'])) in done]
   raw.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in complete_rows))
 attempts_path=OUT/'attempts.jsonl'
 attempt={'attempt_id':f'resume_{len(attempts_path.read_text().splitlines())+1 if attempts_path.exists() else 1:03d}','status':'RUNNING','resumed_completed_instances':len(done),'contract_sha256':CONTRACT_SHA,'code_sha256':{p:hash_file(ROOT/p) for p in ['experiments/WP2/generate_labels.py','experiments/WP2/analyze_labels.py','experiments/WP2/run_experiment.py']},'recorded_at_local':datetime.datetime.now().astimezone().isoformat()}
 with attempts_path.open('a') as f:f.write(json.dumps(attempt,sort_keys=True)+'\n')
 failures=[json.loads(line) for line in failures_file.read_text().splitlines() if line.strip()]
 total=sum(4*len(seeds[sp][int(n)])*len(tasks[sp]) for sp in seeds for n in C['system']['population_sizes'])
 cases=[(sp,task,graph,int(n),int(seed)) for sp in ('train','validation','test') for task in tasks[sp] for graph in C['system']['graph_families'] for n in C['system']['population_sizes'] for seed in seeds[sp][int(n)] if (sp,graph,int(n),task,int(seed)) not in done]
 print(f'resume: {len(done)}/{total} complete; {len(cases)} remaining',flush=True)
 rows=sum(1 for _ in raw.open());completed=len(done)
 for case in cases:
  split,task,graph,n,seed=case; print(f'{completed+1}/{total} {split} {graph} N={n} {task} seed={seed}',flush=True)
  emitted,meta,failure=case_worker(case)
  # Keep partial labels even when this instance fails; a retry uses the same frozen row keys.
  with raw.open('a') as f:
   for row in emitted:f.write(json.dumps(row,sort_keys=True)+'\n')
  rows+=len(emitted)
  if meta is not None:
   with instances.open('a') as f:f.write(json.dumps(meta,sort_keys=True)+'\n')
   done.add(case);completed+=1
  if failure is not None:
   failure['partial_rows_preserved']=len(emitted);failures.append(failure)
   with failures_file.open('a') as f:f.write(json.dumps(failure,sort_keys=True)+'\n')
   print('FAIL '+repr(failure['error']),flush=True)
 failures_out=OUT/'failures.json';failures_out.write_text(json.dumps(failures,indent=2)+'\n')
 manifest={'python':platform.python_version(),'platform':platform.platform(),'numpy':numpy.__version__,'scipy':scipy.__version__,'contract_sha256':CONTRACT_SHA,'scheduler_sha256':SCHED,'code_sha256':{p:hash_file(ROOT/p) for p in ['src/events/ode_segment_scheduler.py','src/events/inputs.py','src/dynamics/tasks.py','src/solvers/oracle.py','experiments/WP2/generate_labels.py','experiments/WP2/analyze_labels.py','experiments/WP2/run_experiment.py']},'rows':rows,'expected_rows':50176,'instances':len(done),'expected_instances':528,'failures':len(failures),'input_seed_splits':{sp:{str(n):ss for n,ss in by_n.items()} for sp,by_n in seeds.items()},'raw_sha256':hash_file(raw),'instances_sha256':hash_file(instances),'attempts_sha256':hash_file(OUT/'attempts.jsonl') if (OUT/'attempts.jsonl').exists() else None,'downstream_science_executed':False}
 (OUT/'labels_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':main()
