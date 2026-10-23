"""Resumable range downloads with authoritative LFS SHA256 verification."""
import os,time,json,hashlib,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests
ROOT=Path('/root/autodl-fs/models/sdxl-refiner-1.0')
REV='5d4cfe854c9a9a87939ff3653551c2b3c99a4356'
FILES=['vae/diffusion_pytorch_model.fp16.safetensors','text_encoder_2/model.fp16.safetensors','unet/diffusion_pytorch_model.fp16.safetensors']
start=time.time()
for name in FILES:
 url=f'https://hf-mirror.com/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/{REV}/{name}'
 response=requests.head(url,allow_redirects=False,timeout=30);response.raise_for_status()
 size=int(response.headers['x-linked-size']);sha=response.headers['x-linked-etag'].strip('"');location=response.headers['location']
 dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True);part=dest.with_suffix('.segmented');state=dest.with_suffix('.segments.json')
 if dest.exists():
  with dest.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else None
  if actual is None:
   h=hashlib.sha256()
   with dest.open('rb') as f:
    for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
   actual=h.hexdigest()
  assert actual==sha,(name,'existing hash mismatch')
  continue
 if state.exists():meta=json.loads(state.read_text())
 else:
  previous=list((ROOT/'.cache/huggingface/download'/Path(name).parent).glob(f'*{sha}.incomplete'))
  prefix=previous[0].stat().st_size if previous else 0
  if previous:os.replace(previous[0],part)
  else:part.touch()
  meta={'prefix':prefix,'done':[],'size':size,'sha':sha};state.write_text(json.dumps(meta))
 ranges=[(s,min(s+16*1024**2,size)-1) for s in range(meta['prefix'],size,16*1024**2)]
 fd=os.open(part,os.O_RDWR);os.ftruncate(fd,size)
 def fetch(pair):
  lo,hi=pair
  for attempt in range(4):
   try:
    with requests.get(location,headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=(20,60)) as r:
     r.raise_for_status();assert r.status_code==206
     assert r.headers.get('Content-Range','').startswith(f'bytes {lo}-{hi}/'),r.headers.get('Content-Range')
     offset=lo
     for block in r.iter_content(1024**2):
      if block:os.pwrite(fd,block,offset);offset+=len(block)
     assert offset==hi+1
    return lo
   except Exception:
    if attempt==3:raise
    time.sleep(2)
 try:
  todo=[p for p in ranges if p[0] not in meta['done']]
  with ThreadPoolExecutor(max_workers=32) as pool:
   futures=[pool.submit(fetch,p) for p in todo]
   for future in as_completed(futures):
    meta['done'].append(future.result());state.write_text(json.dumps(meta))
    if len(meta['done'])%8==0:print(json.dumps({'file':name,'chunks':len(meta['done']),'total_chunks':len(ranges),'elapsed':time.time()-start}),flush=True)
 finally:os.close(fd)
 h=hashlib.sha256()
 with part.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 assert h.hexdigest()==sha,(name,'download hash mismatch')
 os.replace(part,dest);print(json.dumps({'verified':name,'bytes':size,'sha256':sha}),flush=True)
os.environ['HF_ENDPOINT']='https://hf-mirror.com';os.environ['HF_HOME']='/root/autodl-fs/models/hf_cache_sdxl'
from huggingface_hub import snapshot_download
snapshot_download('stabilityai/stable-diffusion-xl-refiner-1.0',revision=REV,local_dir=str(ROOT),allow_patterns=['model_index.json','scheduler/scheduler_config.json','tokenizer_2/*','text_encoder_2/config.json','unet/config.json','vae/config.json'],max_workers=3)
(ROOT/'download_manifest.json').write_text(json.dumps({'repo':'stabilityai/stable-diffusion-xl-refiner-1.0','revision':REV,'weights_sha256_verified':True,'seconds':time.time()-start}))
subprocess.run([sys.executable,'-u','evaluate_sdxl_refiner_pilot.py'],check=True)
