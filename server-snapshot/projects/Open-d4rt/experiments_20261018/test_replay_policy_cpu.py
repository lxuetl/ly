import ast
from pathlib import Path
import unittest
import cv2
import torch
from replay_policy_cpu import ALPHAS,MOTIONS,padded_prefix,key,improves,select,bounded_output,track_prefix


class ReplayPolicyTests(unittest.TestCase):
    def test_padding_has_no_withheld_frames(self):
        rgb=torch.arange(16.).reshape(16,1,1,1).expand(-1,3,8,8)
        p=padded_prefix(rgb[:8]);self.assertEqual(len(p),16)
        self.assertTrue(torch.equal(p[:8],rgb[:1].expand(8,-1,-1,-1)))
        self.assertTrue(torch.equal(p[8:],rgb[:8]))

    def test_padding_rejects_invalid_prefix(self):
        with self.assertRaises(ValueError):padded_prefix(torch.zeros(7,3,8,8))

    def test_future_mutation_does_not_change_prefix_tracks(self):
        torch.manual_seed(20);full=torch.rand(16,3,32,32);mutated=full.clone();mutated[8:]=torch.rand_like(mutated[8:])
        a=track_prefix(full[:8].clone());b=track_prefix(mutated[:8].clone())
        self.assertEqual(a['prefix_sha256'],b['prefix_sha256']);self.assertEqual(a['seed_sha256'],b['seed_sha256'])
        for motion in MOTIONS:self.assertTrue(torch.equal(a['weights'][motion],b['weights'][motion]))
        self.assertTrue(torch.equal(a['uv'],b['uv']))

    def test_no_support_transfer_no_new_writes(self):
        a=track_prefix(torch.rand(8,3,32,32));allowed={m:torch.zeros_like(a['weights'][m],dtype=torch.bool) for m in MOTIONS}
        b=track_prefix(torch.rand(8,3,32,32),None)
        # Use the same prefix/seed count for the actual transfer assertion.
        rgb=torch.rand(8,3,32,32);p=track_prefix(rgb);allowed={m:torch.zeros_like(p['weights'][m],dtype=torch.bool) for m in MOTIONS}
        q=track_prefix(rgb,allowed)
        self.assertFalse(any(v.any() for v in q['weights'].values()))

    def test_LPIPS_harm_rejected_even_when_MSE_improves(self):
        self.assertFalse(improves({'mse':.5,'lpips':1.1},{'mse':1.,'lpips':1.}))

    def test_MSE_harm_rejected_even_when_LPIPS_improves(self):
        self.assertFalse(improves({'mse':1.1,'lpips':.5},{'mse':1.,'lpips':1.}))

    def test_choose_smallest_passing_alpha(self):
        baseline={'mse':1.,'lpips':1.};c={key(m,a):{'mse':.8,'lpips':.9} for m in MOTIONS for a in ALPHAS}
        choice=select(c,baseline,{'hold':1,'cv':1});self.assertTrue(choice.endswith('_a0.125'))
        self.assertEqual(select(c,baseline,{'hold':0,'cv':0}),'keep')

    def test_untested_horizons_bitwise_keep(self):
        base=torch.rand(8,3,8,8);raw=torch.rand_like(base);out=bounded_output(base,raw,.25)
        self.assertTrue(torch.equal(out[4:],base[4:]));self.assertTrue(torch.allclose(out[:4],base[:4]+.25*(raw[:4]-base[:4])))

    def test_blank_prefix_abstains(self):
        result=track_prefix(torch.ones(8,3,32,32)*.5)
        self.assertFalse(any(w.any() for w in result['weights'].values()))

    def test_generation_boundary_and_loader_field_allowlist(self):
        tree=ast.parse(Path(__file__).with_name('run_replay_policy_cpu.py').read_text())
        functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual([a.arg for a in functions['generate'].args.args],['net','rgb','allowed','ungated_control'])
        strings={n.value for n in ast.walk(functions['read_rgb']) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        self.assertTrue({'metadata_json','source_path','images_jpeg_bytes'}<=strings)
        self.assertFalse(any('teacher' in s or 'tracks_' in s or 'visibility' in s for s in strings))


if __name__=='__main__':
    torch.set_num_threads(1);cv2.setNumThreads(1);unittest.main()
