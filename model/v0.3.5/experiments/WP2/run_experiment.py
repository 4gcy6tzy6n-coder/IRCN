"""Fail-closed WP2 run entrypoint; never triggers WP3 or later packages."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from ircn.v03.gates import load_current_statuses
import yaml

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 contract_path=ROOT/'contracts/V03_WP2_INFLUENCE_v1.yaml'; contract=yaml.safe_load(contract_path.read_text()); contract_hash=sha(contract_path)
 statuses=load_current_statuses(ROOT)
 if statuses.get('WP1')!='PASS':raise SystemExit(f"WP2 blocked: WP1={statuses.get('WP1')}")
 if contract.get('status')!='FROZEN_BEFORE_LABEL_GENERATION':raise SystemExit('WP2 contract is not frozen')
 if sha(ROOT/'src/events/ode_segment_scheduler.py')!=contract['system']['candidate_scheduler_sha256']:
  raise SystemExit('candidate scheduler hash differs from frozen WP1 candidate')
 # Validate recorded WP1 evidence and keep all WP1 calibration/confirmation seeds out of WP2.
 cal=yaml.safe_load((ROOT/'contracts/V03_4I_LOCAL_BATCH2_CALIBRATION.yaml').read_text())
 con=yaml.safe_load((ROOT/'contracts/V03_4I_CONFIRMATORY.yaml').read_text())
 cal_result=json.loads((ROOT/'reports/WP1_v034i/calibration.json').read_text())
 con_result=json.loads((ROOT/'reports/WP1_v034i_confirm/confirmatory.json').read_text())
 if cal_result.get('status')!='FULL_CALIBRATION_SCOPE_PASSED' or cal_result.get('completed_instances')!=96 or cal_result.get('runtime_failures')!=0:raise SystemExit('WP1 calibration evidence is incomplete')
 if con_result.get('status')!='FULL_CONFIRMATORY_SCOPE_PASSED' or con_result.get('completed_instances')!=240 or con_result.get('runtime_failures')!=0:raise SystemExit('WP1 confirmatory evidence is incomplete')
 wp1_seeds=set(cal['scope']['calibration_seeds'])|set(cal['scope']['confirmatory_seeds'])
 wp2_seeds=set()
 for field in ('training_seeds_by_n','validation_seeds_by_n','test_seeds_by_n'):
  for values in contract['split'][field].values():wp2_seeds.update(values)
 if wp1_seeds & wp2_seeds:raise SystemExit(f'WP2 reuses WP1 seeds: {sorted(wp1_seeds & wp2_seeds)}')
 out=ROOT/'reports/WP2/run_v1'
 if (out/'analysis.json').exists():raise SystemExit(f'completed analysis already exists; refusing to overwrite: {out}')
 labels_complete=(out/'labels_manifest.json').exists()
 if out.exists() and any(out.iterdir()) and not (out/'labels_raw.jsonl').exists():raise SystemExit(f'partial output is not safely resumable: {out}')
 out.mkdir(parents=True,exist_ok=True)
 status={'work_package':'WP2','status':'RUNNING','contract_sha256':contract_hash,'run_directory':'reports/WP2/run_v1','executed_science':True}
 (ROOT/'reports/WP2/status.json').write_text(json.dumps(status,indent=2)+'\n')
 from generate_labels import main as generate
 from analyze_labels import main as analyze
 try:
  if not labels_complete: generate()
  analyze()
  summary=json.loads((out/'analysis.json').read_text())
  status.update(status=summary['status'],result_file='reports/WP2/run_v1/analysis.json',rows=summary['rows'])
 except BaseException as exc:
  status.update(status='INCONCLUSIVE',failure=repr(exc));raise
 finally:
  (ROOT/'reports/WP2/status.json').write_text(json.dumps(status,indent=2)+'\n')
 print(json.dumps(status,indent=2))
if __name__=='__main__':main()
