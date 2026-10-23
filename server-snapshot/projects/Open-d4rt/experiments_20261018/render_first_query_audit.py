"""Scientific coordinate overlay; observed disagreement selects points, not GT."""
import argparse
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from common import manifest,key
from run_first_query_cpu import OUT,OLD,VideoForecastDataset


def render(index):
    path=manifest()['evaluation'][index]
    old=torch.load(OLD/(key(path)+'_observed.pt'),weights_only=False)
    r=torch.load(OUT/(key(path)+'_identity.pt'),weights_only=False)
    x=VideoForecastDataset('','all',128,paths=[path],explicit_trajectory_interface=True)[0]
    mask=r['memory_mask'][:,-1];score=r['tail_disagreement_px'].masked_fill(~mask,-1)
    points=score.argsort(descending=True)[:4];points=points[mask[points]]
    image=Image.new('RGB',(768,314),'white');draw=ImageDraw.Draw(image)
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    except OSError:font=ImageFont.load_default()
    for column,t in enumerate([0,7,15]):
        rgb=(x['context_rgb'][t].permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
        image.paste(Image.fromarray(rgb).resize((256,256)),(column*256,24))
        draw.text((column*256+5,4),f'Observed frame {t}',fill='black',font=font)
        for i in points.tolist():
            for coords,color,kind in [(old['uv'][i,t],'red','circle'),(r['forward'][i,t],'deepskyblue','square'),(x['history_uv_gt'][old['ids'][i],t],'lime','cross')]:
                if not torch.isfinite(coords).all() or not bool(((coords>=0)&(coords<=1)).all()):continue
                px=column*256+float(coords[0])*255;py=24+float(coords[1])*255
                if kind=='circle':draw.ellipse((px-4,py-4,px+4,py+4),outline=color,width=2)
                elif kind=='square':draw.rectangle((px-5,py-5,px+5,py+5),outline=color,width=2)
                else:
                    draw.line((px-5,py,px+5,py),fill=color,width=2);draw.line((px,py-5,px,py+5),fill=color,width=2)
            draw.text((column*256+4,256-12*points.tolist().index(i)),f'query cache index {int(old["ids"][i])}',fill='white',stroke_width=1,stroke_fill='black',font=font)
    draw.text((5,283),'Red circle: D4RT | Blue square: query-forward LK | Green cross: GT (audit only)',fill='black',font=font)
    draw.text((5,298),'Points: largest observed tail disagreement among complete-loop tracks; illustrative, not random.',fill='black',font=font)
    image.save(OUT/f'case_{index}_coordinates.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,default=0);a=p.parse_args();torch.set_num_threads(1);render(a.case)
