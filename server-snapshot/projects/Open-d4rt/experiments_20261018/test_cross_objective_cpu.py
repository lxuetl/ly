import unittest
import torch
from cross_objective_cpu import common_support,factor_effects,direction_audit,mse_decomposition,MAIN


class CrossObjectiveTests(unittest.TestCase):
    def setup_support(self):
        uv=torch.ones(2,4,2)*.5;old={'uv':uv,'cf':torch.ones(2,4)*.8,'p':torch.ones(1,2,3,2)*.5}
        identity={'local':uv.clone(),'forward':uv.clone(),'memory_mask':torch.ones(2,4,dtype=torch.bool)}
        return old,identity,uv.clone(),torch.ones(2,4,dtype=torch.bool),torch.ones(2,3,2)*.5,torch.ones(2,3,dtype=torch.bool)

    def test_shared_predicted_oracle_bounds_and_visibility(self):
        x=list(self.setup_support());x[0]['p'][0,0,0,0]=1.2;x[4][1,1,0]=-1;x[5][0,2]=False
        _,w,ww=common_support(*x)
        self.assertEqual(int((ww>0).sum()),3);self.assertTrue(torch.allclose(w.sum(-1),torch.ones(2)))

    def test_insufficient_common_history_abstains(self):
        x=list(self.setup_support());x[3][0,2:]=False
        h,_,ww=common_support(*x);self.assertEqual(int(h[0].sum()),2);self.assertFalse((ww[0]>0).any())

    def test_no_frame_weights_without_support(self):
        x=list(self.setup_support());x[1]['memory_mask'][:]=False
        _,w,ww=common_support(*x);self.assertFalse(w.any());self.assertFalse(ww.any())

    def test_factorial_interaction(self):
        values=dict(D_P=10,L_P=12,F_P=11,D_G=7,L_G=6,F_G=5)
        m={a:{k:values[a] for k in ['mse','lpips','temporal']} for a in MAIN}
        result=factor_effects(m);self.assertEqual(result['L_history_at_P']['mse'],2)
        self.assertEqual(result['L_history_at_G']['mse'],-1);self.assertEqual(result['L_interaction']['mse'],-3)

    def test_latent_direction_conflict(self):
        r=direction_audit((torch.tensor([1.]),),(torch.tensor([2.]),))
        self.assertEqual(r['cosine_to_RGB_descent'],-1);self.assertGreater(r['RGB_MSE_directional_derivative_latent'],0)

    def test_MSE_overshoot_is_not_wrong_direction(self):
        r=mse_decomposition(torch.tensor([-2.]),torch.tensor([1.]),torch.tensor([0.]))
        self.assertEqual(r['linear'],-6);self.assertEqual(r['quadratic'],9)
        self.assertEqual(r['delta_MSE_double'],3);self.assertEqual(r['classification'],'finite_step_overshoot')

    def test_MSE_identity_and_wrong_direction(self):
        r=mse_decomposition(torch.tensor([2.]),torch.tensor([1.]),torch.tensor([0.]))
        self.assertEqual(r['identity_residual'],0);self.assertEqual(r['classification'],'wrong_pixel_direction')


if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
