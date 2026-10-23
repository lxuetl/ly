"""Stdlib-only independent arithmetic/protocol checks on downloaded results."""
import ast,hashlib,json,math
from pathlib import Path


def verify(folder):
    folder=Path(folder)
    protocol=json.loads((folder/'protocol.json').read_text())
    assert protocol['formal_training'] is False and protocol['parameter_fitting'] is False
    for name,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((folder.parent/name).read_bytes()).hexdigest()==digest
    tree=ast.parse((folder.parent/'run_reliability_cpu.py').read_text(encoding='utf-8'))
    observed=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='observed')
    used={n.slice.value for n in ast.walk(observed) if isinstance(n,ast.Subscript)
        and isinstance(n.value,ast.Name) and n.value.id=='x' and isinstance(n.slice,ast.Constant)}
    assert used=={'past_uv','past_valid','past_confidence','past_xyz','context_rgb'},used
    rows=json.loads((folder/'history_replay_per_video.json').read_text())
    aggregate=json.loads((folder/'history_replay_summary.json').read_text())
    assert len(rows)==aggregate['videos']==40
    for name,value in aggregate['means'].items():
        expected=sum(r['metrics'][name] for r in rows)/len(rows)
        assert math.isclose(value,expected,rel_tol=1e-8,abs_tol=1e-10),(name,value,expected)
    summary=json.loads((folder/'final_summary.json').read_text())
    assert summary['strict_three_arm_replay_complete'] is False
    for i,row in enumerate(summary['per_case']):
        assert row['index']==i and row['path']==rows[i]['path']
        assert row['weights_updated'] is False and row['TAU_bitwise_verified'] is True
        for name,metrics in row['raw_metrics'].items():
            assert 0<=metrics['budget']<=.002001
            if name!='tau' and not row['support'][name]['accepted_writes']:
                assert metrics['budget']==0
                for metric in ['mse','lpips','temporal']:
                    assert metrics[metric]==row['raw_metrics']['tau'][metric]
        for name,metrics in row['matched_metrics'].items():
            if name=='tau':continue
            original=row['raw_metrics'][name]['budget']
            assert metrics['budget']<=original+1e-8
            if original>1e-10:
                assert abs(metrics['budget']-row['common_positive_budget'])<2e-7
    for kind in ['raw_metrics','matched_metrics']:
        for name,metrics in summary[kind].items():
            for metric,value in metrics.items():
                expected=sum(r[kind][name][metric] for r in summary['per_case'])/4
                assert math.isclose(value,expected,rel_tol=1e-8,abs_tol=1e-10)
    return {'verified':True,'videos':40,'frozen_cases':4,'strict_three_arm_complete':False}


if __name__=='__main__':
    import sys
    print(json.dumps(verify(sys.argv[1]),ensure_ascii=False))
