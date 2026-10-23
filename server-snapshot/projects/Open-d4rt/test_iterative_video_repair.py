import unittest
import torch
import iterative_video_repair as r

class RepairTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3);self.base=torch.rand(8,3,32,32)
        self.sources={'align':torch.rand_like(self.base),'velocity':torch.rand_like(self.base)}
    def test_budget_features_and_modified_state(self):
        st=r.RepairState.start(self.base,budget=.002);c=r.propose(st,self.sources)
        for v in c.values():
            self.assertLessEqual(float((v-self.base).abs().mean()),.002001)
            self.assertEqual(r.features(st,v,self.sources).shape,(27,))
        self.assertTrue(r.execute(st,'align_rgb',c));c2=r.propose(st,self.sources)
        self.assertFalse(torch.equal(c['velocity_rgb'],c2['velocity_rgb']))
        self.assertTrue(r.execute(st,'undo',c2));self.assertTrue(torch.equal(st.current,self.base))
        self.assertFalse(r.execute(st,'align_rgb',r.propose(st,self.sources)))
    def test_abstain_and_safety(self):
        p={'edit':{'gain_lower':1.,'mse_harm_upper':-1.,'temporal_harm_upper':-1.,'baseline_harm_upper':-1.}}
        self.assertEqual(r.choose(p),'keep');self.assertEqual(r.choose(p,calibrated=True),'edit')
        p['edit']['baseline_harm_upper']=.1
        self.assertEqual(r.choose(p,calibrated=True),'keep')
        st=r.RepairState.start(self.base);self.assertFalse(r.execute(st,'keep',r.propose(st,self.sources)))
    def test_max_steps(self):
        st=r.RepairState.start(self.base,max_steps=0)
        self.assertEqual(list(r.propose(st,self.sources)),['keep'])
    def test_scorer_loss_backward(self):
        model=r.PreferenceEvaluator();x=torch.randn(4,27);p=model(x)
        loss=r.preference_loss(p,torch.zeros(4,5),torch.tensor([0,1,0,1]),
                               torch.tensor([[0,1],[2,3]]),torch.ones(5))
        loss.backward();self.assertTrue(torch.isfinite(loss))

if __name__=='__main__':unittest.main()
