# Query-Budget D4RT Research Direction

## Objective

Reduce D4RT decoder queries while preserving downstream accuracy. Encoder cost is fixed; all efficiency claims must therefore report both total latency and decoder-only latency.

## Stage 1: reproducible baselines

Implement and compare `random`, `uniform`, and `task_aware` selectors under identical absolute query budgets. Start with pose and 3D tracking because neither requires dense image output.

Pose budgets count anchor points; the actual decoder call uses `2 * K` queries. Tracking budgets count tracks; the actual decoder call uses `K * T` queries.

## Stage 2: learned task-conditioned policy

Train `TaskConditionedQueryPolicy` with the D4RT encoder and decoder frozen. Candidate features should include encoder features, normalized coordinates, image gradients, motion cues, previous-query confidence, visibility, and coverage. Use a high-budget D4RT run as teacher supervision.

## Stage 3: sequential anytime querying

Split the budget into an exploration batch and one or more refinement batches. Update uncertainty and coverage after every decode. Stop when the task residual, uncertainty, or marginal coverage gain falls below a threshold.

## Required metrics

- Pose: ATE, RPE translation, RPE rotation, Pose AUC.
- Tracking: APD3D, AJ, OA.
- Efficiency: absolute query count, decoder latency, end-to-end latency, peak memory.
- Curves: accuracy vs. query count and accuracy vs. latency.

## Immediate next implementation

1. Add candidate cue extraction from video and cached encoder tokens.
2. Add a pose evaluator that uses `build_pose_queries` and the existing Umeyama solver.
3. Add a tracking evaluator that uses `build_track_queries`.
4. Produce budget sweeps for K in `[8, 16, 32, 64, 128, 256]`.
5. Only after baselines are stable, train the learned query policy.
