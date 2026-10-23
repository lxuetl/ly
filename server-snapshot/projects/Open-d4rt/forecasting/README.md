# D4RT forecasting cache（阶段一与阶段二）

该目录把 D4RT 当作冻结的离线几何教师，不修改其核心网络。

- 阶段一：从前 `C` 帧提取查询点的三维历史、UV、可见性和置信度。
- 阶段二：另跑完整 `C+K` 帧，仅取后 `K` 帧作为教师标签，构造严格的时序训练对。

## 防止未来泄漏

上下文运行和完整视频运行的 D4RT 坐标规范可能不同。程序只使用共同历史帧中的高置信点估计 Sim(3)，然后把**教师未来标签**变换到上下文坐标系；模型输入 `past_xyz` 始终来自只看前 `C` 帧的运行。完整视频运行的历史特征不会写入输入字段。

每个缓存包含 `query_uv`、`past_xyz/past_uv/past_visibility/past_confidence/past_valid`、`future_xyz_teacher/future_visibility/future_confidence/future_valid`，以及对齐参数和 JSON 元数据。

## 运行

先做低显存冒烟测试：

```bash
conda activate d4rt
python forecasting/cache_d4rt_forecasting.py \
  --subsets adt_mini --context-frames 16 --future-frames 8 \
  --num-query-points 16 --query-chunk-size 32 --limit-seqs 1 \
  --output-dir tmp/forecasting_cache_smoke --device cuda
```

正式生成（12GB 显存建议从 128 个查询开始）：

```bash
python forecasting/cache_d4rt_forecasting.py \
  --subsets adt_mini --context-frames 16 --future-frames 8 \
  --num-query-points 128 --query-chunk-size 64 \
  --output-dir data/d4rt_forecasting_cache --device cuda
```

后续轻量轨迹预测器直接读取这些 `.npz`：输入是所有 `past_*` 字段，监督信号是 `future_xyz_teacher` 和 `future_valid`。不要把 `sim3_*` 或完整视频推理结果作为预测时输入。

## 阶段三：轻量未来轨迹预测器

```bash
python forecasting/train_forecaster.py --epochs 100 --batch-size 4
```

模型逐轨迹编码历史几何，再池化场景信息，参数量很小；输出是上下文坐标系中的未来三维轨迹。

## 阶段四：评估与视频模型接口

```bash
python forecasting/evaluate_forecaster.py --split test
```

输出 3D ADE/FDE、常速度基线、轨迹可视化、预测 NPZ 和确定性的 `geometry_features_7d`（相对 XYZ、查询 UV、置信度、时间）。`GeometryConditionEncoder` 将其投影成视频模型的 cross-attention tokens；该投影权重必须与下游视频模型联合训练。

## 阶段五：几何条件视频预测闭环

```bash
python forecasting/train_geometry_video.py --epochs 20 --batch-size 2 --image-size 64
```

阶段三模型保持冻结；预测未来几何按查询 UV 栅格化到视频 latent feature map，并训练空间 adapter 与轻量视频预测器。每轮同时报告有/无几何的验证 L1，便于检查注入是否真正有效。

## 正式阶段五/六：SimVP 注入与公平消融

在 OpenSTL 环境运行：

```bash
source /root/autodl-tmp/envs/OpenSTL/bin/activate
python forecasting/train_simvp_ablation.py --epochs 20 --batch-size 2 --image-size 64
```

脚本以相同随机种子分别训练 `no_geometry`、`history_geometry`、`future_pred_geometry`，在独立测试集输出 MAE、MSE、PSNR、SSIM、权重和预测可视化。SimVP 的视觉上下文为 16 帧，经可学习时间投影得到 8 个未来 latent；空间几何在时空模块之后、解码器之前注入。

Oracle 上限实验单独运行（会合并而非覆盖已有汇总）：

```bash
python forecasting/train_simvp_ablation.py --modes oracle_future_geometry --epochs 20
```

`oracle_future_geometry` 使用看过未来 RGB 的 D4RT 教师标签，只用于诊断注入上限，禁止作为部署输入。

动态栅格化版本使用逐时刻 UV：历史组使用 `past_uv`，预测组从最近四步历史 UV 速度外推，Oracle 使用教师未来 UV；点特征通过双线性四邻域散布。测试指标额外包含动态区域 MAE/PSNR，以及由“相对最后历史帧 RGB 变化超过 0.05”定义的运动掩码 IoU、F1、Precision、Recall。该掩码是运动区域代理，不是人工语义分割标注。

联合 XYZ/UV 轨迹模型与消融：

```bash
python forecasting/train_joint_forecaster.py --epochs 100
python forecasting/train_simvp_ablation.py --modes joint_future_geometry --epochs 20 --output-dir outputs/simvp_dynamic_ablation
```

`joint_future_geometry` 的 XYZ 和逐时刻 UV 来自同一个共享时序编码器，不再使用手工常速度 UV 外推。

多尺度 FiLM 与动态区域加权损失：

```bash
python forecasting/train_simvp_ablation.py --modes joint_future_geometry --injection film --run-name film_joint --epochs 20
python forecasting/train_simvp_ablation.py --modes joint_future_geometry --injection film --dynamic-loss-weight 4 --run-name film_joint_dyn4 --epochs 20
python forecasting/train_simvp_ablation.py --modes oracle_future_geometry --injection film --dynamic-loss-weight 4 --run-name film_oracle_dyn4 --epochs 20
```

FiLM 同时调制低分辨率时空 latent 和高分辨率 decoder skip，并保存两个可学习门控值。动态损失为运动区域像素赋予 `1+λ` 权重。
