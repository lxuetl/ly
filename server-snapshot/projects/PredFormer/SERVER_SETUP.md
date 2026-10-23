# PredFormer server setup

- Repository: https://github.com/yyyujintang/PredFormer
- Commit: a1493c4ade8eed740d7aed11da819e9aa7914c2d
- Checkout: `/root/autodl-tmp/PredFormer`
- Environment: `/root/autodl-tmp/envs/predformer`
- Python 3.10.20; torch 2.6.0+cu124; CUDA runtime12.4.
- Editable OpenSTL package points to this PredFormer checkout.
- This is a lightweight venv with system-site-packages inherited from d4rt. It is NOT a fully standalone environment. Do not delete the d4rt environment. Existing SimVP/TAU files were not removed.
- Tested timm0.9.16 and einops0.8.2; these differ from upstream legacy pins. Import/CLI and synthetic GPU checks passed, not every original dataset pipeline.

## Usage

```bash
source /root/autodl-tmp/envs/predformer/bin/activate
cd /root/autodl-tmp/PredFormer
python tools/train.py --help
python installation_smoke.py
```

## Verified scope

Default Quadruplet TSST: patch8, dim256, heads8, Ndepth3, depth1; 12,731,328 parameters.
RTX3080Ti12GB, AMP, batch1,128x128: input16 frames, native output16, first8 used per official shorter-horizon method. Forward/backward/AdamW update passed with finite gradients; peak allocated1.163GiB. Synthetic step1.56s includes initialization and is not a training epoch ETA.

No pretrained weights downloaded, no real-data training started. Existing data loader and fair split still need explicit integration before experiments. `installation_smoke.json` contains machine-readable results.
