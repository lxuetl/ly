import subprocess,sys
subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir','diffusers==0.35.1','transformers==4.49.0','accelerate==1.4.0','huggingface-hub==0.34.4','safetensors==0.5.3'],check=True)
subprocess.run([sys.executable,'-u','setup_sdxl_refiner.py'],check=True)
