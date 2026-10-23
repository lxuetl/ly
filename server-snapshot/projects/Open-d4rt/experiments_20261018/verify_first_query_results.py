"""Independent stdlib checks of provenance, input boundary and result arithmetic."""
import ast,hashlib,json,math
from pathlib import Path


def verify(folder):
    folder=Path(folder);protocol=json.loads((folder/'protocol.json').read_text())
    assert not protocol['training'] and not protocol['parameter_fitting'] and not protocol['threshold_search']
    for name,digest in protocol['hashes'].items():
        assert hashlib.sha256((folder.parent/name).read_bytes()).hexdigest()==digest
    tree=ast.parse((folder.parent/'first_query_cpu.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='first_query_identity')
    assert [a.arg for a in fn.args.args]==['rgb','query','raw_uv','raw_valid']
    rows=json.loads((folder/'history_per_video.json').read_text());h=json.loads((folder/'history_summary.json').read_text())
    assert len(rows)==h['videos']==40 and [r['index'] for r in rows]==list(range(40))
    for name,value in h['means'].items():
        values=[r['metrics'][name] for r in rows if r['metrics'][name] is not None]
        assert h['nonempty_videos'][name]==len(values)
        if values:assert math.isclose(value,sum(values)/len(values),rel_tol=1e-8,abs_tol=1e-10)
        else:assert value is None
    for name,value in h['pooled_counts'].items():assert value==sum(r['counts'][name] for r in rows)
    c=h['pooled_counts'];assert c['proxy_evaluable']==sum(c[n] for n in ['proxy_true_positive','proxy_false_positive','proxy_false_negative','proxy_true_negative'])
    summary=json.loads((folder/'final_summary.json').read_text())
    for i,row in enumerate(summary['per_case']):
        assert row['index']==i and row['path']==rows[i]['path']
        assert row['TAU_bitwise_verified'] and not row['weights_updated']
        for arm,m in row['raw_metrics'].items():
            assert 0<=m['budget']<=.002001
            if arm=='tau':continue
            actual=row['matched_metrics'][arm]['budget'];assert actual<=m['budget']+1e-8
            if m['budget']>1e-10:assert abs(actual-row['common_positive_budget'])<2e-7
            if not row['support'][arm]['accepted_writes']:
                assert m['budget']==0
                for name in ['mse','lpips','temporal']:assert m[name]==row['raw_metrics']['tau'][name]
        # These arms use precisely the same selected history: only coordinates differ.
        supports=row['support'];assert supports['first_mask_raw']==supports['first_local']==supports['first_forward']
    for kind in ['raw_metrics','matched_metrics']:
        for arm,metrics in summary[kind].items():
            for name,value in metrics.items():
                assert math.isclose(value,sum(r[kind][arm][name] for r in summary['per_case'])/4,rel_tol=1e-8,abs_tol=1e-10)
    paired=json.loads((folder/'paired_anchor_posthoc.json').read_text())
    assert paired['label_only'] and not paired['used_for_generation_or_selection']
    assert len(paired['per_video'])==40
    for name,field in [('first_EPE_mean_same_points','first_EPE_px_same_points'),('last_EPE_mean_same_points','last_EPE_px_same_points')]:
        values=[r[field] for r in paired['per_video'] if r[field] is not None]
        assert math.isclose(paired[name],sum(values)/len(values),rel_tol=1e-8,abs_tol=1e-10)
    for name,value in paired['pooled_counts'].items():assert value==sum(r[name] for r in paired['per_video'])
    endpoints=json.loads((folder/'endpoint_posthoc.json').read_text())
    assert endpoints['label_only'] and not endpoints['used_for_generation_or_selection'] and len(endpoints['cases'])==4
    return {'verified':True,'history_videos':40,'frozen_cases':4,'weights_updated':False}


if __name__=='__main__':
    import sys
    print(json.dumps(verify(sys.argv[1]),ensure_ascii=False))
