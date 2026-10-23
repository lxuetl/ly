"""Posthoc arithmetic only: no model calls, hyperparameter selection or deployment."""
import json,math,sys
from pathlib import Path


def main(folder):
    folder=Path(folder);summary=json.loads((folder/'final_summary.json').read_text());rows=summary['per_case']
    arms=list(rows[0]['diagnostics']);main_arms=['D_P','L_P','F_P','D_G','L_G','F_G']
    statistics={}
    for a in arms:
        diagnostics=[r['diagnostics'][a] for r in rows]
        statistics[a]={'own_objective_decreases':sum(d['matched']['own_objective_after']<d['own_objective_before'] for d in diagnostics),
            'latent_cosine_mean':sum(d['cosine_to_RGB_descent'] for d in diagnostics)/4,
            'latent_cosine_negative':sum(d['cosine_to_RGB_descent']<0 for d in diagnostics),
            'MSE_worse_than_TAU':sum(r['matched_metrics'][a]['mse']>r['matched_metrics']['tau']['mse'] for r in rows),
            'finite_step_harm_cases':[i for i,d in enumerate(diagnostics) if d['matched']['pixel_MSE_decomposition']['classification']=='finite_step_overshoot'],
            'wrong_pixel_direction_cases':[i for i,d in enumerate(diagnostics) if d['matched']['pixel_MSE_decomposition']['classification']=='wrong_pixel_direction']}
    sensitivities=[]
    for r in rows:
        decompositions={a:r['diagnostics'][a]['matched']['pixel_MSE_decomposition'] for a in arms}
        minimum=min(decompositions[a]['quadratic'] for a in main_arms)
        # Alternative L2 matching; it does NOT also preserve equal L1 budgets.
        equal_l2={a:math.sqrt(minimum/max(decompositions[a]['quadratic'],1e-100))*decompositions[a]['linear']+minimum for a in main_arms}
        quarter={a:.25*d['linear']+.0625*d['quadratic'] for a,d in decompositions.items()}
        sensitivities.append({'index':r['index'],'same_L2_delta_MSE':equal_l2,'quarter_residual_delta_MSE':quarter,
            'same_L2_history_effect':{h+'_'+endpoint:equal_l2[h+'_'+endpoint]-equal_l2['D_'+endpoint] for h in ['L','F'] for endpoint in ['P','G']}})
    result={'label_only_posthoc':True,'used_for_action_or_parameter_selection':False,
        'statistics':statistics,'sensitivities':sensitivities,
        'quarter_residual_definition':'Y(alpha)=TAU+alpha*(matched_output-TAU), fixed alpha=.25; MSE exact by pixel identity; no latent reoptimization, no LPIPS claim',
        'equal_L2_definition':'scale down the six primary matched outputs to common squared L2 magnitude; alternative to L1 matching, not simultaneous L1 and L2 matching',
        'history_target_vs_true_future_MSE_mean':{h:sum(r['history_target_vs_true_future_MSE'][h] for r in rows)/4 for h in ['D','L','F','H']}}
    (folder/'diagnostic_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'statistics':statistics,'sensitivity_cases':len(sensitivities)},ensure_ascii=False))


if __name__=='__main__':main(sys.argv[1])
