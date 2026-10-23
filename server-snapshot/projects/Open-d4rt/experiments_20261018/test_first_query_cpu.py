import unittest
from unittest.mock import patch
import numpy as np
import torch
import first_query_cpu as f


class FirstQueryTests(unittest.TestCase):
    def setup_case(self,drift=0):
        rgb=torch.rand(16,3,32,32);query=torch.tensor([[.25,.25]])
        truth=query[:,None].expand(-1,16,-1).clone();truth[0,:,0]+=torch.arange(16)*.2/31
        uv=truth.clone();uv[0,-1,0]+=drift/31
        return rgb,query,uv,torch.ones(1,16,dtype=torch.bool)

    def mocked(self,drift=0,fail=False):
        x=self.setup_case(drift)
        def lk(a,b,p):
            delta=float(b[0,0])-float(a[0,0]);q=np.asarray(p).copy();q[:,0]+=delta*.2
            ok=np.ones(len(p),bool)
            if fail and int(a[0,0])==3 and int(b[0,0])==4:ok[:]=False
            return q,ok,np.zeros(len(p))
        def patches(rgb,uv):return torch.linspace(0,1,25).expand(uv.shape[0],uv.shape[1],25)
        with patch.object(f,'gray_video',return_value=[np.full((32,32),i,np.uint8) for i in range(16)]),patch.object(f,'lk',lk),patch.object(f,'patches',patches):
            return x,f.first_query_identity(*x)

    def test_correct_identity_passes(self):
        _,r=self.mocked();self.assertTrue(r['identity_mask'].all());self.assertFalse(r['proxy_drift'].any())

    def test_wrong_last_anchor_is_detected(self):
        _,r=self.mocked(5)
        self.assertTrue(r['proxy_drift'].all());self.assertFalse(r['identity_mask'].any())
        self.assertTrue(r['memory_mask'].all())
        self.assertAlmostEqual(float(r['D4RT_return_error_px'][0]),5,places=4)

    def test_local_repair_is_bounded(self):
        (_,_,uv,_),r=self.mocked(5)
        distance=((r['local']-uv)*31).norm(dim=-1)
        self.assertLessEqual(float(distance.max()),2.00001)
        self.assertAlmostEqual(float(distance[0,-1]),2,places=4)

    def test_failed_track_never_resurrects(self):
        _,r=self.mocked(fail=True)
        self.assertFalse(r['forward_valid'][:,4:].any());self.assertFalse(r['memory_mask'].any())

    def test_uniform_query_texture_abstains(self):
        x=list(self.setup_case());x[0]=torch.ones_like(x[0])*.5
        r=f.first_query_identity(*x);self.assertFalse(r['memory_mask'].any())


if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
