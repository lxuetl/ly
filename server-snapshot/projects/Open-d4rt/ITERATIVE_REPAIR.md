# Preference-driven iterative repair: diagnostic v1

Frozen TAU -> immutable history-derived sources -> current-state proposals ->
shared preference/risk scorer -> calibrated reject/accept/undo -> next state.
No D4RT, diffusion editor, backbone training, or TTPO reproduction claim.

## Implemented

- `iterative_video_repair.py`: five forward candidates (keep plus two sources
  times RGB/texture edits), exact previous-state undo, three-round maximum,
  baseline-relative L1 budget, repeat-state blocking, 27-dimensional label-free
  features, shared 64-wide MLP, supervised relative gain/risk and ranking loss.
- `build_iterative_repair_diagnostic.py`: deterministic random modified-state
  rollouts, then separately computed LPIPS/MSE/temporal supervision; writes
  features and labels separately and keeps original-video grouping.
- `test_iterative_video_repair.py`: budget, rollback, cycle, stop, candidate
  regeneration, loss backward and uncalibrated-abstention checks.

## Deliberate limits

The diagnostic builder reads original validation caches, not a training split.
DO NOT train the scorer on those records and claim validation generalization.
Use disjoint original training videos for fit, separate held-out groups for
normalization/calibration, and untouched groups for modified-state evaluation.
Random rollouts include potentially harmful edits independent of truth; safe
states alone would hide rollback failures. A future audit must report initial
versus edited states separately, safe acceptance precision/recall, harmful
acceptance, beneficial rollback recall, false rollback, abstention and per-round
quality with original-video bootstrap. Keep identity is an abstention, not a
positive repair. Undo returns exactly to a stored prior state, but whether that
is a useful repair must be measured, never assumed.

Sources are cached; residual proposals depend on the edited current state, but
motion is not re-estimated. The spatial region is the maximum-disagreement tile
shared across horizons, not an object-tracked tube. Features are global summary
statistics and may miss local semantic errors. This is a cheap falsifiable first
scorer, not a claim of semantic reasoning. Calibration margins are not supplied
and must not be fabricated; uncalibrated inference abstains by default.

Budget .025 and edit strength .25 are prototype configuration values, not tuned
or endorsed deployment parameters. Preserve old experiments unchanged. First
establish ranking on modified states before full iterative rollouts. Match total
candidate evaluations and cumulative correction budget against a one-shot
selector. Do not tune on TUM or historical test results.
