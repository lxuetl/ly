import unittest
import importlib.machinery
import importlib.util
from pathlib import Path
import torch
from trajectory_attention import TrackCrossAttention,sample,splat
from visibility_interface import future_gate

class VisibilityTests(unittest.TestCase):
    def test_real_dataset_raw_labels_and_padding(self):
        from common import manifest
        from video_forecast_dataset import VideoForecastDataset,video_pad_collate
        import copy
        path=manifest()['dev'][0]
        old=VideoForecastDataset('','all',128,paths=[path])[0]
        new=VideoForecastDataset('','all',128,paths=[path],explicit_trajectory_interface=True)[0]
        self.assertTrue(torch.equal(old['context_rgb'],new['context_rgb']))
        self.assertTrue(torch.equal(old['future_rgb'],new['future_rgb']))
        self.assertTrue(torch.allclose(old['future_uv_teacher'],new['future_uv_raw'].clamp(0,1),equal_nan=True))
        uv=new['future_uv_raw'];inside=torch.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<=1).all(-1)
        self.assertTrue(torch.equal(inside,new['future_in_bounds']))
        small=copy.deepcopy(new)
        for k,v in small.items():
            if torch.is_tensor(v) and k not in ['context_rgb','future_rgb']:small[k]=v[:-1]
        batch=video_pad_collate([new,small])
        self.assertFalse(batch['future_visibility_gt'][1,-1].any())
        self.assertFalse(batch['future_in_bounds'][1,-1].any())
        with self.assertRaises(ValueError):video_pad_collate([old,new])
        with self.assertRaises(ValueError):video_pad_collate([new,old])

    def test_legacy_numerical_compatibility(self):
        path=Path(__file__).with_name('trajectory_attention.py.pre_visibility_v2')
        if not path.exists():self.skipTest('server-side original backup unavailable')
        loader=importlib.machinery.SourceFileLoader('legacy_trajectory_attention',str(path))
        spec=importlib.util.spec_from_loader(loader.name,loader);legacy=importlib.util.module_from_spec(spec);loader.exec_module(legacy)
        torch.manual_seed(9);old=legacy.TrackCrossAttention(4);new=TrackCrossAttention(4)
        torch.nn.init.normal_(old.out.weight,std=.1);new.load_state_dict(old.state_dict())
        args=(torch.randn(1,2,4,8,8),torch.randn(1,2,3,4),torch.rand(1,2,3,2),torch.ones(1,2,3,dtype=torch.bool),torch.rand(1,2,3),torch.rand(1,2,2,2))
        a,sa=old(*args);b,sb=new(*args)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(sa,sb))

    def test_provenance_and_bounds(self):
        uv=torch.tensor([[[[.5,.5],[1.1,.5],[float('nan'),.5]]]])
        self.assertEqual(future_gate(uv).tolist(),[[[1.,0.,0.]]])
        with self.assertRaises(ValueError):future_gate(uv,torch.ones(1,1,3),source='oracle_gt')
        with self.assertRaises(ValueError):future_gate(uv,torch.ones(1,1,3),source='unknown')
        self.assertEqual(future_gate(uv,torch.ones(1,1,3),source='oracle_gt',oracle_diagnostic=True).tolist(),[[[1.,0.,0.]]])

    def test_visible_identity_and_horizon_gate(self):
        torch.manual_seed(42);m=TrackCrossAttention(4);torch.nn.init.normal_(m.out.weight,std=.1)
        h=torch.randn(1,2,4,8,8);vals=torch.randn(1,2,3,4);uv=torch.rand(1,2,3,2);fu=torch.rand(1,2,2,2)
        valid=torch.ones(1,2,3,dtype=torch.bool);conf=torch.full((1,2,3),.5)
        old,s=m(h,vals,uv,valid,conf,fu)
        one,s1=m(h,vals,uv,valid,conf,fu,future_visibility=torch.ones(1,2,2),visibility_source='predicted')
        self.assertTrue(torch.equal(old,one));self.assertTrue(torch.equal(s,s1))
        vis=torch.tensor([[[0.,1.],[0.,1.]]])
        out,s=m(h,vals,uv,valid,conf,fu,future_visibility=vis,visibility_source='oracle_gt',oracle_diagnostic=True)
        self.assertTrue(torch.equal(out[:,0],h[:,0]));self.assertFalse(s[:,0].any())

    def test_nonfinite_and_all_hidden(self):
        m=TrackCrossAttention(4);m.out.bias.data.fill_(1)
        h=torch.randn(1,2,4,8,8);vals=torch.full((1,1,3,4),float('nan'));uv=torch.full((1,1,3,2),.5)
        out,s=m(h,vals,uv,torch.ones(1,1,3,dtype=torch.bool),torch.ones(1,1,3),torch.full((1,1,2,2),float('nan')))
        self.assertTrue(torch.equal(out,h));self.assertFalse(s.any())
        self.assertTrue(torch.isfinite(sample(h,torch.full((1,1,2,2),float('nan')))).all())
        z,ss=splat(torch.full((1,1,2,4),float('nan')),torch.full((1,1,2,2),.5),torch.ones(1,1,2),8,8)
        self.assertFalse(z.any());self.assertFalse(ss.any())

if __name__=='__main__':unittest.main()
