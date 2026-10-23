"""Descriptive, video-paired uncertainty for the prelocked eight-case diagnostic."""
import json,random,statistics,argparse
from pathlib import Path

def main(root):
    rows=[json.loads((root/f'case_{i}.json').read_text()) for i in range(8)]
    geom=[json.loads((root/f'geometry_{i}.json').read_text()) for i in range(8)]
    rng=random.Random(20261020)
    samples=[rng.choices(range(8),k=8) for _ in range(20000)]
    def estimate(v):
        b=sorted(statistics.mean(v[i] for i in ids) for ids in samples)
        return {'mean':statistics.mean(v),'descriptive_CI95':[b[500],b[19499]],'positive_videos':sum(x>0 for x in v)}
    effects={m:{k:estimate([r['matched_attribution'][m][k] for r in rows]) for k in
        ['history_shapley','endpoint_shapley','third_stage_total','appearance_target_gap','fusion_objective_gap','interaction']}
        for m in ['mse','lpips','temporal']}
    summary=json.loads((root/'final_summary.json').read_text());mets=summary['matched_metrics'];base=mets['tau']
    relative={a:{k:100*(v[k]/base[k]-1) for k in ['mse','lpips','temporal']} for a,v in mets.items()}
    shares={m:{k:100*summary['matched_attribution'][m][k]/summary['matched_attribution'][m]['total_recoverable_vs_RGB_GT']
        for k in ['history_shapley','endpoint_shapley','third_stage_total','appearance_target_gap','fusion_objective_gap']}
        for m in ['mse','lpips']}
    conditional={m:{name:estimate([r['matched_metrics'][a][m]-r['matched_metrics'][b][m] for r in rows])
        for name,(a,b) in {'history_fix_at_pred':('D_P','H_P'),'history_fix_at_GT':('D_G','H_G'),
        'endpoint_fix_at_D4RT':('D_P','D_G'),'endpoint_fix_at_GT_history':('H_P','H_G')}.items()} for m in ['mse','lpips']}
    geometry={'pred_minus_CV_ADE':estimate([g['metrics']['UV_ADE_px128']['predicted']-g['metrics']['UV_ADE_px128']['cv_model'] for g in geom]),
        'GT_history_reduces_future_RGB_MAE_videos':sum(g['metrics']['history_G_to_true_future_RGB_MAE']<g['metrics']['history_D_to_true_future_RGB_MAE'] for g in geom)}
    result={'scope':'descriptive eight previously evaluated held-out test videos, not fresh or full-dataset confirmation',
        'effects_positive_mean_recoverable_loss_decrease':effects,'relative_to_TAU_percent_negative_better':relative,
        'signed_share_of_DP_to_RGB_GT_gap_percent_NOT_share_of_total_prediction_error':shares,
        'conditional_fixes_positive_better':conditional,'geometry':geometry}
    (root/'quantification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root)
