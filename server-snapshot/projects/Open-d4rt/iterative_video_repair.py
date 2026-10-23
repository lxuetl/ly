"""Stateful repair primitives. No target tensors or quality labels in inference API.

Sources are immutable history-derived RGB proposals, not future ground truth.
Cached-source pilot recomputes residuals, NOT optical flow, after each edit.
"""
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F


def low(x):
    return F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode='replicate'), 5, 1)


@dataclass
class RepairState:
    baseline: torch.Tensor
    current: torch.Tensor
    stack: list = field(default_factory=list)
    steps: int = 0
    max_steps: int = 3
    budget: float = .025
    visited: list = field(default_factory=list)

    @classmethod
    def start(cls, baseline, **kwargs):
        assert baseline.ndim == 4 and torch.isfinite(baseline).all()
        assert baseline.min() >= 0 and baseline.max() <= 1
        b = baseline.detach().clone()
        return cls(b, b.clone(), visited=[b.clone()], **kwargs)


def project(x, state):
    delta = x.clamp(0, 1) - state.baseline
    scale = min(1., state.budget / max(float(delta.abs().mean()), 1e-12))
    return state.baseline + scale * delta


@torch.no_grad()
def propose(state, sources, alpha=.25):
    """One shared space-time region per round, selected without target labels.

    Source disagreement selects a diagnostic region, not a certified risk mask.
    Same source, region and alpha support RGB versus high-frequency proposals.
    """
    if state.steps >= state.max_steps:
        return {'keep': state.current.clone()}
    assert sources
    for source in sources.values():
        assert source.shape == state.current.shape and torch.isfinite(source).all()
    disagreement = torch.stack([(v-state.current).abs().mean((0, 1)) for v in sources.values()]).mean(0)
    h, w = disagreement.shape
    tile = min(32, h, w)
    pooled = F.avg_pool2d(disagreement[None, None], tile, tile)[0, 0]
    index = int(pooled.argmax()); y, x = divmod(index, pooled.shape[1])
    mask = torch.zeros_like(disagreement)
    mask[y*tile:(y+1)*tile, x*tile:(x+1)*tile] = 1
    mask = F.avg_pool2d(F.pad(mask[None, None], (4,4,4,4), mode='replicate'), 9, 1)
    candidates = {'keep': state.current.clone()}
    for name, source in sorted(sources.items()):
        delta = source-state.current
        candidates[name+'_rgb'] = project(state.current+alpha*mask*delta, state)
        candidates[name+'_texture'] = project(state.current+alpha*mask*(delta-low(delta)), state)
    if state.stack:
        candidates['undo'] = state.stack[-1].clone()
    return candidates


def _stats(x):
    temporal = x[1:]-x[:-1]
    return torch.stack([x.abs().mean(), x.square().mean().sqrt(),
                        (x-low(x)).abs().mean(), temporal.abs().mean()])


FEATURE_VERSION = 'relative-state-v1-27'
class PreferenceEvaluator(torch.nn.Module):
    """Shared small scorer. Outputs gain, step MSE/time, base MSE/time, safe logit.

    Feature normalization buffers MUST be fitted on training videos only.
    Raw outputs are predictions, not confidence bounds; calibration is separate.
    """
    def __init__(self, width=64):
        super().__init__()
        self.register_buffer('mean',torch.zeros(27))
        self.register_buffer('scale',torch.ones(27))
        self.net=torch.nn.Sequential(torch.nn.Linear(27,width),torch.nn.SiLU(),
                                     torch.nn.Linear(width,width),torch.nn.SiLU(),
                                     torch.nn.Linear(width,6))

    def forward(self,x):
        return self.net((x-self.mean)/self.scale.clamp_min(1e-6))


def preference_loss(pred, targets, safe, pair_indices, target_scale):
    """Pairs must belong to the SAME current state, never across videos/states.

    targets: [N,5] offline deltas; safe: [N]; pair_indices: [P,2].
    Ranking target puts safe repairs above unsafe edits, then ranks gain.
    target_scale is fit on training data only to balance metric magnitudes.
    """
    regression=F.smooth_l1_loss(pred[:,:5]/target_scale,targets/target_scale)
    safety=F.binary_cross_entropy_with_logits(pred[:,5],safe.float())
    ranking=pred.sum()*0
    if pair_indices.numel():
        i,j=pair_indices.unbind(1)
        preference=torch.where(safe[i]!=safe[j],safe[i].float()-safe[j].float(),
                               (targets[i,0]-targets[j,0]).sign())
        valid=preference!=0
        if valid.any():
            ranking=F.softplus(-preference[valid]*(pred[i[valid],5]-pred[j[valid],5])).mean()
    return regression+safety+ranking


@torch.no_grad()
def calibrated_predictions(model,state,candidates,sources,margins):
    """Margins supplied by disjoint-video calibration, not guessed in inference.

    Five nonnegative error margins; empirical calibration is not a guarantee
    under domain shift. Without a calibration artifact caller must abstain.
    """
    assert margins.shape==(5,) and torch.isfinite(margins).all() and (margins>=0).all()
    names=list(candidates)
    prediction=model(torch.stack([features(state,candidates[n],sources) for n in names]))
    results={}
    for n,p in zip(names,prediction):
        results[n]={'gain_lower':float(p[0]-margins[0]),
                    'mse_harm_upper':float(p[1]+margins[1]),
                    'temporal_harm_upper':float(p[2]+margins[2]),
                    'baseline_harm_upper':float(torch.maximum(p[3]+margins[3],p[4]+margins[4]))}
    return results


def features(state, candidate, sources):
    """No action/family IDs, no targets. Shared candidate descriptor, 27 dimensions."""
    residual = candidate-state.current
    descriptors = [_stats(state.current-state.baseline), _stats(residual),
                   _stats(candidate-state.baseline)]
    # Symmetric source descriptors avoid source-family ID shortcuts.
    src_stats = torch.stack([_stats(candidate-source) for source in sources.values()])
    descriptors += [src_stats.mean(0), src_stats.min(0).values, src_stats.max(0).values]
    descriptors.append(candidate.new_tensor([state.steps/max(state.max_steps,1),
                                           len(state.stack)/max(state.max_steps,1),
                                           float((residual.abs().mean(1) > 1e-5).float().mean())]))
    return torch.cat(descriptors)


def choose(predictions, calibrated=False, gain_margin=0., risk_limit=0.):
    """Predictions map action -> conservative gain lower / harm upper bounds.

    Bounds must be calibrated on held-out original videos. This module does not
    claim a calibration guarantee. Uncalibrated evaluators always abstain.
    """
    if not calibrated:
        return 'keep'
    eligible = {n: p['gain_lower'] for n,p in predictions.items()
                if n != 'keep' and p['gain_lower'] > gain_margin
                and p['mse_harm_upper'] <= risk_limit
                and p['temporal_harm_upper'] <= risk_limit
                and p['baseline_harm_upper'] <= risk_limit}
    return max(eligible, key=eligible.get) if eligible else 'keep'


def execute(state, action, candidates):
    """Reject is a terminal keep. Undo restores exactly one accepted state."""
    if state.steps >= state.max_steps or action == 'keep':
        return False
    target = candidates[action]
    if action != 'undo' and any(torch.allclose(target, p, atol=1e-7, rtol=0) for p in state.visited):
        return False
    assert float((target-state.baseline).abs().mean()) <= state.budget+1e-6
    if action == 'undo':
        assert state.stack and torch.equal(target,state.stack[-1])
        state.stack.pop()
    else:
        state.stack.append(state.current.clone())
    state.current = target.clone(); state.steps += 1; state.visited.append(target.clone())
    return True
