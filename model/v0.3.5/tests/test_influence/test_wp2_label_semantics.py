import numpy as np
from dynamics.system import System
from experiments.WP2.generate_labels import features_at, integrate_window

class ConstantInput:
    def __call__(self,t): return np.array([0.0])
    def breakpoints(self,a,b): return []

def test_zero_coupling_skip_changes_only_after_release_and_label_is_finite():
    s=System(W=np.zeros((2,2)),U=np.zeros((2,1)),b=np.array([.2,-.1]),tau=np.ones(2))
    x=ConstantInput(); h=np.array([.1,-.1]); base,_=integrate_window(s,x,h,0.)
    skip,_=integrate_window(s,x,h,0.,clamp_node=0)
    assert np.array_equal(base[0],skip[0])
    assert np.allclose(skip[1,0],h[0])
    assert np.isfinite(np.sqrt(np.mean((base-skip)**2)))

def test_feature_calculation_uses_present_past_state_and_is_finite():
    s=System(W=np.array([[0.,.2],[.1,0.]]),U=np.zeros((2,1)),b=np.zeros(2),tau=np.ones(2))
    f,_=features_at(s,ConstantInput(),.1,np.array([.2,-.1]))
    assert f.shape==(2,8)
    assert np.isfinite(f).all()

def test_analysis_pipeline_smoke_with_synthetic_rows(tmp_path, monkeypatch):
    import json, hashlib
    from pathlib import Path
    import experiments.WP2.analyze_labels as analysis
    out=tmp_path/'WP2';out.mkdir()
    monkeypatch.setattr(analysis,'OUT',out)
    rows=[]
    for split,seeds,tasks in [('train',[16001],['smooth','sparse_pulse','dense_burst']),('validation',[16011],['smooth','sparse_pulse','dense_burst']),('test',[16021,16022],['smooth','state_switch'])]:
        for seed in seeds:
            for task in tasks:
                for graph in ['chain','star','feedback_ring','modular_sparse']:
                    for node in range(16):
                        influence=float(node+1 + ((seed+node)%3)*.1)
                        feat=[float(node),float(node%4),float(node)/100.,.2,.3,float(node)/10.,1.,0.]
                        rows.append({'split':split,'seed':seed,'task':task,'graph':graph,'n':16,'snapshot_time':.1,'node':node,'influence':influence,'per_time_difference_rms':[influence]*21,'per_time_baseline_rms':[1.]*21,'contract_sha256':hashlib.sha256((analysis.ROOT/'contracts/V03_WP2_INFLUENCE_v1.yaml').read_bytes()).hexdigest(),'features':feat,'feature_extract_seconds':1e-5,'local_update_seconds':1e-3})
    raw=out/'labels_raw.jsonl';raw.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'failures.json').write_text('[]')
    (out/'instances.jsonl').write_text('')
    (out/'labels_manifest.json').write_text(json.dumps({'contract_sha256':hashlib.sha256((analysis.ROOT/'contracts/V03_WP2_INFLUENCE_v1.yaml').read_bytes()).hexdigest(),'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}))
    monkeypatch.setattr(analysis,'coverage_check',lambda rows:{'passed':True,'expected_rows':len(rows),'observed_rows':len(rows),'duplicate_rows':0,'missing_rows':0,'unexpected_rows':0})
    monkeypatch.setattr(analysis,'expected_row_count',lambda:len(rows))
    analysis.main()
    result=json.loads((out/'analysis.json').read_text())
    assert result['status']=='INCONCLUSIVE'  # fixture omits the full frozen instance/accuracy log
    assert (out/'metrics.csv').exists()

def test_local_node_update_cost_probe_accepts_snapshot_time():
    from experiments.WP2.generate_labels import local_node_update_seconds
    s=System(W=np.zeros((1,1)),U=np.zeros((1,1)),b=np.array([.1]),tau=np.ones(1))
    elapsed=local_node_update_seconds(s,ConstantInput(),np.array([0.]),.1,0)
    assert np.isfinite(elapsed) and elapsed>0
