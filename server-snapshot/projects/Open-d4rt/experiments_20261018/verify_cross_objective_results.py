"""Independent stdlib verification of matched factorial and MSE decomposition."""
import hashlib,json,math
from pathlib import Path

ARMS=['D_P','L_P','F_P','D_G','L_G','F_G','H_G','T_G','RGB_GT']


def effects(m):
    result={}
    for h in ['L','F']:
        result[h+'_history_at_P']={k:m[h+'_P'][k]-m['D_P'][k] for k in ['mse','lpips','temporal']}
        result[h+'_history_at_G']={k:m[h+'_G'][k]-m['D_G'][k] for k in ['mse','lpips','temporal']}
        result[h+'_interaction']={k:m[h+'_G'][k]-m[h+'_P'][k]-m['D_G'][k]+m['D_P'][k] for k in ['mse','lpips','temporal']}
    for h in ['D','L','F']:result[h+'_endpoint_G_minus_P']={k:m[h+'_G'][k]-m[h+'_P'][k] for k in ['mse','lpips','temporal']}
    return result


def verify(folder):
    folder=Path(folder);p=json.loads((folder/'protocol.json').read_text());summary=json.loads((folder/'final_summary.json').read_text())
    assert p['CPU_only'] and not p['training'] and not p['parameter_fitting'] and not p['threshold_search']
    assert p['all_arms']==ARMS and p['cases']==[0,1,2,3]
    for name,digest in p['hashes'].items():assert hashlib.sha256((folder.parent/name).read_bytes()).hexdigest()==digest
    assert len(summary['per_case'])==4
    for i,r in enumerate(summary['per_case']):
        assert r['index']==i and r['TAU_bitwise_verified'] and not r['weights_updated']
        assert r['support']['same_support_for_all_feature_arms'] and r['support']['future_writes']>0
        for kind in ['raw','matched']:
            m=r[kind+'_metrics'];expected=effects(m)
            for effect,values in expected.items():
                for metric,value in values.items():assert abs(r['effects_'+kind][effect][metric]-value)<1e-12
            for a in ARMS:
                budget=m[a]['budget'];assert 0<budget<=.002001
                if kind=='matched':assert abs(budget-r['common_positive_budget'])<2e-7
                diag=r['diagnostics'][a];d=diag[kind]['pixel_MSE_decomposition']
                assert abs(d['linear']+d['quadratic']-d['delta_MSE_double'])<1e-12
                assert abs(d['identity_residual'])<1e-12 and d['quadratic']>=0
                assert abs((m[a]['mse']-m['tau']['mse'])-d['delta_MSE_double'])<1e-8
                assert -1.000001<=diag['cosine_to_RGB_descent']<=1.000001
        assert all(r['matched_metrics'][a]['budget']<=r['raw_metrics'][a]['budget']+1e-8 for a in ARMS)
        # Pairwise effects use the changed output as delta relative to comparator.
        for effect in ['L','F']:
            for endpoint in ['P','G']:
                measured=r['pairwise_matched_MSE_decomposition'][effect+'_at_'+endpoint]
                expected=r['effects_matched'][effect+'_history_at_'+endpoint]['mse']
                assert abs(measured['delta_MSE_double']-expected)<1e-8
    for kind in ['raw_metrics','matched_metrics']:
        for a,values in summary[kind].items():
            for metric,value in values.items():
                assert math.isclose(value,sum(r[kind][a][metric] for r in summary['per_case'])/4,rel_tol=1e-9,abs_tol=1e-12)
    for effect,values in summary['effects_matched'].items():
        for metric,value in values.items():assert abs(value-sum(r['effects_matched'][effect][metric] for r in summary['per_case'])/4)<1e-12
    post=json.loads((folder/'diagnostic_summary.json').read_text())
    assert post['label_only_posthoc'] and not post['used_for_action_or_parameter_selection']
    for a in ARMS:
        observed=sum(r['diagnostics'][a]['matched']['own_objective_after']<r['diagnostics'][a]['own_objective_before'] for r in summary['per_case'])
        assert post['statistics'][a]['own_objective_decreases']==observed
    for r,sensitivity in zip(summary['per_case'],post['sensitivities']):
        assert r['index']==sensitivity['index']
        ds={a:r['diagnostics'][a]['matched']['pixel_MSE_decomposition'] for a in ARMS}
        minimum=min(ds[a]['quadratic'] for a in ARMS[:6])
        for a in ARMS:
            expected=.25*ds[a]['linear']+.0625*ds[a]['quadratic']
            assert abs(sensitivity['quarter_residual_delta_MSE'][a]-expected)<1e-15
        for a in ARMS[:6]:
            expected=math.sqrt(minimum/ds[a]['quadratic'])*ds[a]['linear']+minimum
            assert abs(sensitivity['same_L2_delta_MSE'][a]-expected)<1e-15
    return {'verified':True,'CPU_only':True,'cases':4,'arms_per_case':9,'weights_updated':False}


if __name__=='__main__':
    import sys
    print(json.dumps(verify(sys.argv[1]),ensure_ascii=False))
