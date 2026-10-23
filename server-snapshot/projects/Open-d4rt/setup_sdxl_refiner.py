"""Download only SDXL refiner FP16 components to the data disk."""
import os,json,time
from pathlib import Path
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HOME']='/root/autodl-fs/models/hf_cache_sdxl'
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='60'
from huggingface_hub import snapshot_download
root=Path('/root/autodl-fs/models/sdxl-refiner-1.0')
revision='5d4cfe854c9a9a87939ff3653551c2b3c99a4356'
patterns=['model_index.json','scheduler/scheduler_config.json','tokenizer_2/*',
 'text_encoder_2/config.json','text_encoder_2/model.fp16.safetensors',
 'unet/config.json','unet/diffusion_pytorch_model.fp16.safetensors',
 'vae/config.json','vae/diffusion_pytorch_model.fp16.safetensors']
start=time.time()
snapshot_download('stabilityai/stable-diffusion-xl-refiner-1.0',revision=revision,local_dir=str(root),allow_patterns=patterns,max_workers=3)
report={'repo':'stabilityai/stable-diffusion-xl-refiner-1.0','revision':revision,'endpoint':os.environ['HF_ENDPOINT'],'seconds':time.time()-start,'files':{str(p.relative_to(root)):p.stat().st_size for p in root.rglob('*') if p.is_file() and '.cache' not in str(p)}}
(root/'download_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
