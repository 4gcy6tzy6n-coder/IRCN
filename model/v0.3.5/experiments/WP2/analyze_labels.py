"""Fit prespecified WP2 influence proxy and analyze untouched test split."""
from __future__ import annotations
import csv, json, math, time, hashlib
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'reports/WP2/run_v1'; C=yaml.safe_load((ROOT/'contracts/V03_WP2_INFLUENCE_v1.yaml').read_text())
FEATURES=C['features']['candidate']; ALPHAS=C['model']['ridge_alpha_grid']; RNG_SEED=C['statistics']['confidence_interval']['rng_seed']

def load_rows():
 rows=[]
 with (OUT/'labels_raw.jsonl').open() as f:
  for line in f:
   z=json.loads(line); z['features']=np.asarray(z['features'],float); z['influence']=float(z['influence'])
   if z.get('contract_sha256')!=hashlib.sha256((ROOT/'contracts/V03_WP2_INFLUENCE_v1.yaml').read_bytes()).hexdigest():raise ValueError('row contract hash mismatch')
   diff=np.asarray(z['per_time_difference_rms'],float); base=np.asarray(z['per_time_baseline_rms'],float)
   reconstructed=float(np.sqrt(np.mean(diff**2))/max(float(np.sqrt(np.mean(base**2))),1e-12))
   if diff.shape!=(21,) or base.shape!=(21,) or not np.isfinite(z['features']).all() or not np.isfinite(diff).all() or not np.isfinite(base).all() or not math.isclose(reconstructed,z['influence'],rel_tol=1e-10,abs_tol=1e-14):
    raise ValueError(f"raw influence integrity check failed for {z.get('graph')}/{z.get('n')}/{z.get('task')}/{z.get('seed')}/{z.get('node')}")
   rows.append(z)
 return rows

def grouped(rows):
 g=defaultdict(list)
 for r in rows:g[(r['split'],r['graph'],int(r['n']),r['task'],int(r['seed']),float(r['snapshot_time']))].append(r)
 return g

def binary_truth(items):
 order=sorted(range(len(items)),key=lambda j:(-items[j]['influence'],int(items[j]['node'])))
 k=max(1,math.ceil(.2*len(items))); y=np.zeros(len(items),dtype=int);y[order[:k]]=1
 return y,order,k

def ranking_metrics(items,scores):
 y,ideal,k=binary_truth(items); order=sorted(range(len(items)),key=lambda j:(-float(scores[j]),int(items[j]['node'])))
 hit=sum(y[j] for j in order[:k]); recall=float(hit/max(1,y.sum()))
 rel=np.asarray([r['influence'] for r in items]); dcg=sum(float(rel[j])/math.log2(rank+2) for rank,j in enumerate(order[:k])); idcg=sum(float(rel[j])/math.log2(rank+2) for rank,j in enumerate(ideal[:k])); ndcg=float(dcg/idcg) if idcg else 1.0
 ap=0.; hits=0
 for rank,j in enumerate(order,1):
  if y[j]:hits+=1;ap+=hits/rank
 ap=float(ap/max(1,int(y.sum())))
 rho=float(spearmanr([r['influence'] for r in items],scores).statistic) if len(items)>1 and np.std([r['influence'] for r in items])>0 and np.std(scores)>0 else 0.0
 if not math.isfinite(rho):rho=0.0
 return {'recall_at_20':recall,'ndcg_at_20':ndcg,'average_precision':ap,'spearman':rho,'y':y,'order':order}

def fit_ridge(X,y,alpha):
 mean=X.mean(axis=0); sd=X.std(axis=0);sd[sd<1e-12]=1.
 z=(X-mean)/sd; ym=float(y.mean()); beta=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-ym))
 return {'mean':mean,'sd':sd,'ym':ym,'beta':beta},lambda x:((np.asarray(x)-mean)/sd)@beta+ym

def metric_mean(rows,score_func,split):
 groups=grouped([r for r in rows if r['split']==split and r['task']!='state_switch']); vals=[]
 for key,items in groups.items():vals.append(ranking_metrics(items,score_func(items))['recall_at_20'])
 return float(np.mean(vals)) if vals else float('nan')

def stable_group_seed(key):
 import hashlib
 return int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:4],'big')

def baseline_scores(name,items,random_seed=0):
 if name=='activity':return np.asarray([r['features'][0] for r in items])
 if name=='local_error':return np.asarray([r['features'][2] for r in items])
 if name=='degree':return np.asarray([r['features'][3]+r['features'][4] for r in items])
 if name=='random':return np.random.default_rng(random_seed).random(len(items))
 raise KeyError(name)

def fit_platt(scores,y):
 scores=np.asarray(scores,float);y=np.asarray(y,float)
 def loss(z):
  p=expit(z[0]*scores+z[1]);return float(np.sum(np.logaddexp(0,z[0]*scores+z[1])-y*(z[0]*scores+z[1])))
 res=minimize(loss,np.array([1.,0.]),method='L-BFGS-B')
 if not res.success:raise RuntimeError('Platt calibration failed: '+res.message)
 return res.x

def calibration_metrics(p,y):
 order=np.argsort(p); bins=np.array_split(order,min(10,len(order)));ece=0.
 for b in bins:
  if len(b):ece+=len(b)/len(y)*abs(float(np.mean(p[b]))-float(np.mean(y[b])))
 return {'brier':float(np.mean((p-y)**2)),'ece_10_equal_count':float(ece)}

def expected_row_count():
 seed_key={'train':'training_seeds_by_n','validation':'validation_seeds_by_n','test':'test_seeds_by_n'}
 task_map={'train':C['split']['train_validation_tasks'],'validation':C['split']['train_validation_tasks'],'test':C['split']['test_tasks']}
 total=0
 for split,key in seed_key.items():
  for n in C['system']['population_sizes']:
   sampled=min(C['sampling']['training_nodes_per_snapshot'],int(n)) if split=='train' else int(n)
   total+=len(C['system']['graph_families'])*len(task_map[split])*len(C['split'][key][int(n)])*len(C['sampling']['snapshot_times_seconds'])*sampled
 return total

def coverage_check(rows):
 # Script execution places experiments/WP2 on sys.path, not necessarily the
 # repository root; resolve the shared sampler from the repository explicitly.
 if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
 from experiments.WP2.generate_labels import selected_nodes
 observed=set();duplicates=0
 for r in rows:
  key=(r['split'],r['graph'],int(r['n']),r['task'],int(r['seed']),float(r['snapshot_time']),int(r['node']))
  if key in observed:duplicates+=1
  observed.add(key)
 expected=set(); seed_key={'train':'training_seeds_by_n','validation':'validation_seeds_by_n','test':'test_seeds_by_n'}
 task_map={'train':C['split']['train_validation_tasks'],'validation':C['split']['train_validation_tasks'],'test':C['split']['test_tasks']}
 for split,seed_field in seed_key.items():
  for task in task_map[split]:
   for graph in C['system']['graph_families']:
    for n in C['system']['population_sizes']:
     for seed in C['split'][seed_field][int(n)]:
      for t in C['sampling']['snapshot_times_seconds']:
       for node in selected_nodes(graph,int(n),task,int(seed),float(t),split):expected.add((split,graph,int(n),task,int(seed),float(t),int(node)))
 return {'expected_rows':len(expected),'observed_rows':len(observed),'duplicate_rows':duplicates,'missing_rows':len(expected-observed),'unexpected_rows':len(observed-expected),'passed':expected==observed and duplicates==0}

def main():
 rows=load_rows(); groups=grouped(rows)
 if not rows:raise RuntimeError('no labels present')
 raw_path=OUT/'labels_raw.jsonl'; manifest=json.loads((OUT/'labels_manifest.json').read_text())
 if manifest.get('contract_sha256')!=hashlib.sha256((ROOT/'contracts/V03_WP2_INFLUENCE_v1.yaml').read_bytes()).hexdigest():raise ValueError('dataset manifest contract hash mismatch')
 raw_hash=hashlib.sha256(raw_path.read_bytes()).hexdigest()
 if manifest.get('raw_sha256')!=raw_hash:raise ValueError('dataset raw file hash mismatch')
 coverage=coverage_check(rows)
 train=[r for r in rows if r['split']=='train']; val=[r for r in rows if r['split']=='validation'];test=[r for r in rows if r['split']=='test']
 Xtr=np.stack([r['features'] for r in train]); ytr=np.log1p([r['influence'] for r in train]); Xv=np.stack([r['features'] for r in val]); yv=np.log1p([r['influence'] for r in val])
 training_start=time.perf_counter();alpha_results=[]; best=None
 for alpha in ALPHAS:
  m,predict=fit_ridge(Xtr,ytr,float(alpha)); rec=metric_mean(val,lambda items:predict(np.stack([r['features'] for r in items])),'validation')
  alpha_results.append({'alpha':alpha,'validation_recall_at_20':rec})
  if best is None or rec>best[0]+1e-12 or (abs(rec-best[0])<=1e-12 and alpha>best[1]):best=(rec,float(alpha))
 # Retain the train-only model so validation-fitted Platt calibration stays aligned.
 fit,predict=fit_ridge(Xtr,ytr,best[1])
 # Validation chooses one cheap comparator; random uses mean across 100 frozen replicates.
 base_names=['activity','local_error','degree','random']; val_scores={}
 val_groups={k:v for k,v in groups.items() if k[0]=='validation' and k[3]!='state_switch'}
 for name in base_names:
  vv=[]
  for key,items in val_groups.items():
   if name=='random':rs=np.mean([ranking_metrics(items,baseline_scores(name,items,88021+i))['recall_at_20'] for i in range(100)])
   else:rs=ranking_metrics(items,baseline_scores(name,items))['recall_at_20']
   vv.append(rs)
  val_scores[name]=float(np.mean(vv))
 baseline=max(base_names,key=lambda z:(val_scores[z],z))
 # Platt scaling is fitted on validation scores only.
 val_pred=[];val_y=[]
 for key,items in val_groups.items():
  scores=predict(np.stack([r['features'] for r in items])); met=ranking_metrics(items,scores); val_pred.extend(scores.tolist());val_y.extend(met['y'].tolist())
 platt=fit_platt(np.asarray(val_pred),np.asarray(val_y)); training_seconds=time.perf_counter()-training_start
 test_groups={k:v for k,v in groups.items() if k[0]=='test'}
 result_rows=[]; per_cluster=defaultdict(lambda:defaultdict(list)); cost_ratios=[]
 for key,items in test_groups.items():
  _,graph,n,task,seed,snapshot=key
  cs=predict(np.stack([r['features'] for r in items])); cm=ranking_metrics(items,cs)
  if task=='state_switch': category='ood'
  else:category='in_distribution'
  scores={'candidate':cs,'activity':baseline_scores('activity',items),'local_error':baseline_scores('local_error',items),'degree':baseline_scores('degree',items)}
  random_recall=[]; random_ndcg=[]; random_ap=[]
  for rep in range(100):
   rm=ranking_metrics(items,baseline_scores('random',items,88021+rep+stable_group_seed(key)))
   random_recall.append(rm['recall_at_20']);random_ndcg.append(rm['ndcg_at_20']);random_ap.append(rm['average_precision'])
  scores['random_expected']=None
  for name,ss in scores.items():
   if name=='random_expected':
    met={'recall_at_20':float(np.mean(random_recall)),'ndcg_at_20':float(np.mean(random_ndcg)),'average_precision':float(np.mean(random_ap)),'spearman':float('nan')}
   else:met=ranking_metrics(items,ss)
   row={'graph':graph,'n':n,'task':task,'seed':seed,'snapshot_time':snapshot,'category':category,'method':name,**{k:v for k,v in met.items() if k not in ('y','order')}}
   result_rows.append(row)
   if category=='in_distribution':
    cluster=(n,task,seed);per_cluster[cluster][name].append(met['recall_at_20'])
  if baseline=='random':
   selected_metric={'recall_at_20':float(np.mean(random_recall)),'ndcg_at_20':float(np.mean(random_ndcg)),'average_precision':float(np.mean(random_ap)),'spearman':float('nan')}
  else:selected_metric=ranking_metrics(items,scores[baseline])
  result_rows.append({'graph':graph,'n':n,'task':task,'seed':seed,'snapshot_time':snapshot,'category':category,'method':'selected_baseline',**{k:v for k,v in selected_metric.items() if k not in ('y','order')}})
  if category=='in_distribution':per_cluster[(n,task,seed)]['selected_baseline'].append(selected_metric['recall_at_20'])
  prob=expit(platt[0]*cs+platt[1]);cal=calibration_metrics(prob,cm['y'])
  result_rows.append({'graph':graph,'n':n,'task':task,'seed':seed,'snapshot_time':snapshot,'category':category,'method':'candidate_calibration',**cal})
  # Estimate break-even using local update costs of nodes not selected at frozen budget.
  _,_,k=binary_truth(items); chosen=set(cm['order'][:k]); saved=sum(float(r['local_update_seconds']) for j,r in enumerate(items) if j not in chosen)
  feature_cost=max(float(r['feature_extract_seconds']) for r in items)
  start=time.perf_counter()
  for _ in range(20):
   trial=predict(np.stack([r['features'] for r in items]));sorted(range(len(items)),key=lambda j:(-float(trial[j]),int(items[j]['node'])))
  score_cost=(time.perf_counter()-start)/20
  cost_ratios.append({'graph':graph,'n':n,'task':task,'seed':seed,'snapshot_time':snapshot,'feature_seconds':feature_cost,'scoring_seconds':score_cost,'saved_local_update_seconds':saved,'cost_ratio':(feature_cost+score_cost)/saved if saved>0 else float('inf')})
 diffs=[]
 for cl,metrics in per_cluster.items():
  a=np.mean(metrics['candidate']);b=np.mean(metrics['selected_baseline']);diffs.append(float(a-b))
 if len(diffs)<2:raise RuntimeError('insufficient independent test clusters for bootstrap')
 rng=np.random.default_rng(RNG_SEED);d=np.asarray(diffs);boot=np.empty(int(C['statistics']['confidence_interval']['resamples']))
 for i in range(len(boot)):boot[i]=np.mean(d[rng.integers(0,len(d),len(d))])
 ci=np.quantile(boot,[.025,.975]).tolist()
 ind_cost=[r for r in cost_ratios if r['task']!='state_switch' and np.isfinite(r['cost_ratio'])]
 predictor_costs=[r['feature_seconds']+r['scoring_seconds'] for r in ind_cost]
 saved_costs=[r['saved_local_update_seconds'] for r in ind_cost]
 median_predictor_cost=float(np.median(predictor_costs));median_saved_cost=float(np.median(saved_costs));median_cost_ratio=median_predictor_cost/median_saved_cost if median_saved_cost>0 else float('inf')
 total_saved_test=float(sum(r['saved_local_update_seconds'] for r in ind_cost));training_payback_ratio=training_seconds/total_saved_test if total_saved_test>0 else float('inf')
 instance_meta=[json.loads(line) for line in (OUT/'instances.jsonl').open() if line.strip()]
 label_generation_wall_seconds=float(sum(z.get('label_generation_wall_seconds',0.) for z in instance_meta))
 accuracy_limit=float(C['system']['feature_state_accuracy_nrmse_limit'])
 accuracy_bad=[z for z in instance_meta if not np.isfinite(z.get('candidate_full_trace_nrmse',float('inf'))) or z['candidate_full_trace_nrmse']>accuracy_limit]
 accuracy_summary={'instances':len(instance_meta),'expected_instances':528,'nrmse_limit':accuracy_limit,'max_nrmse':max((z.get('candidate_full_trace_nrmse',float('inf')) for z in instance_meta),default=None),'pass':len(instance_meta)==528 and not accuracy_bad}
 # OOD descriptive only.
 summary={'contract_sha256':__import__('hashlib').sha256((ROOT/'contracts/V03_WP2_INFLUENCE_v1.yaml').read_bytes()).hexdigest(),'status':('INCONCLUSIVE' if json.loads((OUT/'failures.json').read_text()) or len(rows)!=expected_row_count() or not coverage['passed'] or not accuracy_summary['pass'] else ('PASS' if ci[0]>0 and median_predictor_cost<median_saved_cost and training_seconds<total_saved_test else 'FAIL')),'rows':len(rows),'expected_rows':expected_row_count(),'dataset_integrity':coverage,'feature_state_accuracy':accuracy_summary,'alpha_selection':{'selected':best[1],'validation_recall_at_20':best[0],'grid':alpha_results},'validation_baseline_recall_at_20':val_scores,'selected_baseline':baseline,'test_primary':{'in_distribution_clusters':len(diffs),'mean_candidate_minus_baseline_recall_at_20':float(np.mean(d)),'ci95_cluster_bootstrap':ci,'bootstrap_replicates':len(boot),'bootstrap_rng_seed':RNG_SEED},'test_ood_state_switch':{'clusters':len({(int(r['n']),int(r['seed'])) for r in test if r['task']=='state_switch'}),'candidate_recall_at_20':float(np.mean([r['recall_at_20'] for r in result_rows if r.get('method')=='candidate' and r.get('category')=='ood']))},'cost':{'median_predictor_cost_seconds':median_predictor_cost,'median_saved_local_update_cost_seconds':median_saved_cost,'ratio_of_medians':median_cost_ratio,'median_snapshot_cost_ratio':float(np.median([r['cost_ratio'] for r in ind_cost])),'break_even_pass':bool(median_predictor_cost<median_saved_cost and training_seconds<total_saved_test),'total_saved_local_update_cost_in_distribution_seconds':total_saved_test,'training_payback_ratio':training_payback_ratio,'label_generation_wall_seconds':label_generation_wall_seconds},'training_wall_seconds':training_seconds,'candidate_model':{'algorithm':'numpy ridge regression','target':'log1p influence','alpha':best[1],'feature_mean':fit['mean'].tolist(),'feature_sd':fit['sd'].tolist(),'beta':fit['beta'].tolist(),'intercept':fit['ym']},'platt_calibration':{'slope':float(platt[0]),'intercept':float(platt[1])},'calibration':{'brier':float(np.mean([r['brier'] for r in result_rows if r.get('method')=='candidate_calibration' and r.get('category')=='in_distribution'])),'ece_10_equal_count':float(np.mean([r['ece_10_equal_count'] for r in result_rows if r.get('method')=='candidate_calibration' and r.get('category')=='in_distribution']))},'failures':json.loads((OUT/'failures.json').read_text()),'e2_e3_e4_run':False}
 with (OUT/'metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in result_rows for k in r)));w.writeheader();w.writerows(result_rows)
 with (OUT/'costs.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(cost_ratios[0]));w.writeheader();w.writerows(cost_ratios)
 (OUT/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
 (OUT/'bootstrap_cluster_differences.csv').write_text('cluster_index,recall_difference\n'+'\n'.join(f'{i},{x:.12g}' for i,x in enumerate(d))+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
