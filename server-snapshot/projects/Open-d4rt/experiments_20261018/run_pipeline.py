"""Serial GPU pipeline with explicit completion/failure artifacts; no formal run."""
import os,subprocess,sys,time,traceback
from pathlib import Path
from common import OUT,PROJECT,dump,sha


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).resolve().parent;start=time.time()
    stages=[('tests','test_prototypes.py'),('trajectory','run_trajectory.py'),('contract','audit_before_training.py'),('local_risk','run_local_risk.py')]
    dump(OUT/'RUNNING.json',{'pid':os.getpid(),'started':start,'stages':stages,'script_hashes':{p.name:sha(p) for p in script.glob('*.py')}})
    try:
        for name,file in stages:
            if name in ['trajectory','local_risk'] and (OUT/name/'COMPLETE.json').exists():continue
            dump(OUT/'pipeline_stage.json',{'stage':name,'time':time.time()})
            subprocess.run([sys.executable,'-u',str(script/file)],cwd=str(PROJECT),check=True)
        dump(OUT/'COMPLETE.json',{'seconds':time.time()-start,'formal_training':False,'test_used':False,'TUM_used':False})
        print('PIPELINE_COMPLETE',flush=True)
    except Exception:
        dump(OUT/'FAILED.json',{'time':time.time(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
