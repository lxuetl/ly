"""Final source/data provenance audit; no model execution or training."""
import json,unittest
from pathlib import Path
import torch
from common import manifest,sha,dump,PROJECT
from run_alignment_v2 import DEST,VideoForecastDataset,load,device,inputs

def main():
    torch.set_num_threads(4);m=manifest();paths=m['dev']+m['evaluation'];ds=VideoForecastDataset('','all',128,paths=paths,explicit_trajectory_interface=True)
    for i,path in enumerate(paths):
        x=ds[i];d=device(load(path));fu,fv=inputs(path,x,d)
        assert torch.allclose(fu.clamp(0,1),d['oracle'],atol=2e-6,rtol=0,equal_nan=True),path
    # Dataset and attention integration tests cover collate/provenance/no-op behavior.
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern='test*.py')
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful()
    protocol=json.loads((DEST/'protocol.json').read_text())
    for name,expected in protocol['hashes'].items():assert sha(Path(__file__).with_name(name))==expected
    source_paths=[PROJECT/'forecasting/video_forecast_dataset.py']+[Path(__file__).with_name(n) for n in ['trajectory_attention.py','visibility_interface.py','run_alignment_v2.py','run_budgetmatched_v2.py','test_visibility_v2.py','audit_alignment_integrity.py']]
    dump(DEST/'interface_integrity.json',{'checked_videos':len(paths),'raw_uv_clipped_equals_original_oracle_all_videos':True,'original_run_source_hashes_match':True,'tests_run':result.testsRun,'tests_passed':True,'weights_updated':False,'hashes':{str(p):sha(p) for p in source_paths}})
    print('FINAL_INTERFACE_INTEGRITY_PASS',len(paths),flush=True)

if __name__=='__main__':main()
