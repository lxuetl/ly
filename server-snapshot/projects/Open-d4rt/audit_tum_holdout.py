import json,os,hashlib,time
from pathlib import Path
import numpy as np,cv2
ROOT=Path('/root/autodl-fs/d4rt_training');OUT=ROOT/'tum_holdout_audit_20261017'
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2))
def frames(p):
 with np.load(p,allow_pickle=True) as z:
  if 'images_jpeg_bytes' not in z:return
  for b in z['images_jpeg_bytes']:
   arr=np.frombuffer(b,dtype=np.uint8) if isinstance(b,(bytes,bytearray)) else np.asarray(b,dtype=np.uint8)
   im=cv2.imdecode(arr,cv2.IMREAD_COLOR)
   if im is None:raise ValueError(str(p))
   yield im
def sig(im):
 x=cv2.resize(im,(128,128),interpolation=cv2.INTER_AREA);gray=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY);small=cv2.resize(gray,(9,8));dh=np.packbits(small[:,1:]>small[:,:-1]).tobytes().hex()
 return hashlib.sha256(x.tobytes()).hexdigest(),dh,float(gray.std())
def main():
 OUT.mkdir(exist_ok=True);start=time.time();tum=sorted(Path('data/worldtrack_release/tum').glob('*.npz'));assert len(tum)==8
 exact={};percept={};counts={}
 for p in tum:
  n=0
  for i,im in enumerate(frames(p)):
   sha,dh,std=sig(im);exact.setdefault(sha,[]).append([p.name,i]);
   if std>10:percept.setdefault(dh,[]).append([p.name,i])
   n+=1
  counts[p.name]=n
 names={p.stem.rsplit('_',1)[0]:p.name for p in tum};hits=[];scanned=0;errors=[]
 for directory,dirs,files in os.walk(ROOT):
  dirs[:]=[d for d in dirs if d not in ['cache','__pycache__'] and not any(x in d for x in ['tum_holdout','video_tradeoff_inventory'])]
  for name in files:
   if not name.endswith(('.json','.jsonl','.log','.txt','.csv')):continue
   p=Path(directory)/name
   try:
    with p.open(errors='replace') as f:
     for line in f:
      for needle,seq in names.items():
       if needle in line:hits.append({'file':str(p),'sequence':seq});break
    scanned+=1
   except OSError as e:errors.append(str(e))
 sources=set()
 for root in [Path('data/worldtrack_release'),Path('/root/autodl-fs/d4rt_datasets')]:
  for directory,dirs,files in os.walk(root):
   dirs[:]=[d for d in dirs if d not in ['rgb','rgbs','images','depth','depths','normals','masks','tum'] and len(Path(directory).relative_to(root).parts)<5]
   if 'worldtrack' not in directory:continue
   for name in files:
    if name.endswith('.npz'):sources.add(str((Path(directory)/name).resolve()))
 matches=[];near=[];nf=0
 for j,p in enumerate(sorted(sources)):
  try:
   for i,im in enumerate(frames(p)):
    sha,dh,std=sig(im);nf+=1
    if sha in exact:matches.append({'source':p,'frame':i,'tum':exact[sha]})
    elif std>10 and dh in percept:near.append({'source':p,'frame':i,'tum':percept[dh]})
  except Exception as e:errors.append({'source':p,'error':str(e)})
  if (j+1)%100==0:print(json.dumps({'sources_done':j+1,'total':len(sources),'frames':nf,'seconds':time.time()-start}),flush=True)
 save('summary.json',{'tum_frames':counts,'record_files_scanned':scanned,'history_name_hits':hits,'source_files':len(sources),'frames_checked':nf,'decoded128_exact_matches':matches,'dhash_equal_candidates':near,'errors':errors,'seconds':time.time()-start,'scope':'retained training records recursively; worldtrack RGB archives under current repository and dataset root <=5directory levels. Excludes prior inventory report. Not deleted records or arbitrary renamed/rescaled/cropped near duplicates; equal dHash is only review candidate.'});print('AUDIT_COMPLETE',flush=True)
if __name__=='__main__':main()
