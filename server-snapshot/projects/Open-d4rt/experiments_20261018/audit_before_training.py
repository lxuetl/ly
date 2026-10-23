"""Readonly old-score contract audit, saved separately from existing artifacts."""
import json
from pathlib import Path
from collections import Counter
from common import ROOT,OUT,dump


def main():
    p=ROOT/'iterative_preference_v1_20261017'
    calibration=json.loads((p/'calibration.json').read_text())
    summary=json.loads((p/'summary.json').read_text());counts={}
    for f in sorted((p/'records').glob('val-*.json')):
        for row in json.loads(f.read_text()):
            c=counts.setdefault(row['step'],Counter());c['states']+=1
            labels=row['labels'].values()
            c['available_at_1pct']+=any(z['safe_improvement'] for z in labels)
            c['available_at_zero']+=any(z['gain']>1e-4 and all(z[k]<=0 for k in ['mse_delta','temporal_delta','baseline_mse_delta','baseline_temporal_delta']) for z in labels)
    result={'old_calibration':calibration,'old_decisions':summary,'availability':counts,
      'correction':'new fit uses identical .01 relative safety tolerance for labeling and decisions, separate metric units, rank on actual gain head',
      'remaining_unknown':'calibration remains empirical and may abstain; no safety guarantee under domain shift'}
    dump(OUT/'local_risk/G0_contract_audit.json',result)
    print('G0_CONTRACT_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
