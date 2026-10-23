# 2026-10-18 两条冻结主干原型

服务器：connect.westb.seetacloud.com:34220；密码只通过当前进程环境提供，不写入文件。
代码：/root/autodl-tmp/Open-d4rt/experiments_20261018
结果：/root/autodl-fs/d4rt_training/trajectory_lhcr_pilot_20261018

数据盘稳定检查：三次只读遍历，时间戳1792329923/1792330028/1792330056，文件数均106606，逻辑字节45896389574/45896389574/45896393670。最后4096字节差异对应.autodl/autopanel.monitor.db；无复制或恢复用户进程，关键缓存/权重可读取。此后才创建原型目录和上传代码。

执行顺序：

1. 七项原型不变量测试：采样坐标、精确no-op、无效点回退、边界、预算不放大、评分/风险，以及静止轨迹下固定位置/轨迹注意力等价（包含逐horizon特征预算的一致性）。
2. 冻结TAU128与轨迹预测器；每条原视频仅取一个片段，训练分区沿用旧scorer的fit/dev/calibration原视频分组。最多fit64、dev12、calibration12、evaluation40（原validation，已复用，非新鲜测试）。不读取test/TUM样本。
3. 单层轨迹时间Cross-Attention：Oracle未来UV、预测未来UV、预测轨迹均匀可靠性聚合、固定未来位置检索历史特征。仅训练新增适配器6轮，使用相同初始化/损失/修改预算，以dev含epoch0选择。Oracle只替换UV，不读取未来可见性。记录Oracle UV原缓存clamp限制。历史appearance与固定位置memory为fp16缓存，TAU hidden/skip与基线float32保持精确no-op。
4. G0旧标签/阈值口径只读审计。新标签和推理统一1%相对MSE/temporal非劣容差，LPIPS gain绝对值，直接训练实际排序的gain头。
5. 单轮原历史传播候选，同tile/alpha、共享.002全局L1预算和.03逐像素cap，对比重新训练的27维全局scorer与局部时空CNN。局部输入不接受target；无效/未知visibility不伪装为真值。scorer最多30短轮；dev选权重，独立calibration估计每指标margin，evaluation不调阈值。
6. 对照keep、dev选定固定velocity、校准global/local、明确标为诊断的无校准global/local，以及与local同来源/同编辑类型/同实际修改量的均匀空间对照。报告接受率、错误接受、有益修改覆盖、收益排序和多指标视频级配对CI。

固定参数均为低成本探针，不是正式训练结论。若Oracle无收益，先查接口；Oracle有收益而预测无收益，先查未来UV；attention不胜均匀聚合时保留简单方法。校准全keep也如实报告，不通过evaluation放宽门槛制造正结果。全局scorer和局部scorer参数量接近但不完全相等，以parameter_counts.json为准。

run_pipeline.py使用独立子进程串行占用GPU，进度、完成和失败均写JSON；nohup使任务不随SSH断开结束。不自动启动正式训练，也不同时铺开Omega/TTT/热核。

实际运行环境为 `/root/miniconda3/envs/d4rt/bin/python`（PyTorch2.6.0+cu124）；基础conda环境缺lightning，首次启动后已切回项目环境。失败/重启日志保留在结果目录attempts中，未删除旧实验。

两项补充诊断均保留独立目录：`trajectory_sensitivity`对冻结适配器作0/1/4/16倍注入幅度检查，仅dev选择系数、不再训练；`local_risk_normalized`用同一缓存、划分、30轮与目标函数增加fit-only输出归一化头，同时重跑global/local。补充诊断由主对照结果引出，属于事后探索，不能作为独立确认。

所有实验现已完成；结果和适配器下载于本地results/。当前结论与限制见`结果分析.md`，未启动正式训练。
