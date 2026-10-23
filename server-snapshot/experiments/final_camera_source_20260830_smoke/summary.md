# CaPSR v3 geometry counterfactual audit

Test sequences: 2

Positive degradation means the condition is worse than D4RT UV+XYZ.

| Condition | MSE | MSE degradation (95% CI) | LPIPS degradation (95% CI) | Temporal degradation (95% CI) |
|---|---:|---:|---:|---:|
| base_only | 0.008636 | +0.001702 [+0.001702, +0.001702] | +nan [+nan, +nan] | +0.003174 [+0.003174, +0.003174] |
| d4rt_uv_xyz | 0.006934 | reference | reference | reference |
| d4rt_uv_only | 0.006934 | -0.000001 [-0.000001, -0.000001] | +nan [+nan, +nan] | -0.000001 [-0.000001, -0.000001] |
| affine_2d | 0.006919 | -0.000016 [-0.000016, -0.000016] | +nan [+nan, +nan] | -0.000211 [-0.000211, -0.000211] |
| homography_2d | 0.006904 | -0.000030 [-0.000030, -0.000030] | +nan [+nan, +nan] | -0.000208 [-0.000208, -0.000208] |
| raft_small_2d | 0.006918 | -0.000017 [-0.000017, -0.000017] | +nan [+nan, +nan] | -0.000119 [-0.000119, -0.000119] |
