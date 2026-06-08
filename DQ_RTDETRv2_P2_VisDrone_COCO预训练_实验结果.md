# DQ RT-DETRv2 P2 VisDrone COCO 预训练实验结果记录

创建时间：2026-06-04 10:35:05 CST

工程路径：

```bash
/home/wang.kui/proj/RTDETR/DQ_RTDETRv2_P2
```

训练代码路径：

```bash
/home/wang.kui/proj/RTDETR/DQ_RTDETRv2_P2/rtdetrv2_pytorch
```

COCO 预训练权重：

```bash
/home/wang.kui/proj/RTDETR/RT-DETR_copy copy/rtdetrv2_r50vd_6x_coco_ema.pth
```

当前后台任务：

```bash
screen -r dq_rtdetrv2_p2_a2_a4
```

## 实验设置

| 编号 | 实验名称 | 配置文件 | 主要改动 | 状态 |
| --- | --- | --- | --- | --- |
| A0 | P2 baseline | `configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml` | 原始 RT-DETRv2-R50VD + P2，无 DQ 模块 | 已完成 |
| A1 | Count loss | `configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a1_count.yml` | 加入 CCM 计数分类分支和 `loss_count`，不使用 CGFE，不使用动态 query | 已完成 |
| A2 | CGFE | `configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a2_cgfe.yml` | 加入 CCM + CGFE，不使用动态 query | 已完成 |
| A3 | Dynamic query | `configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a3_dynamic_query.yml` | 加入 CCM + `loss_count` + 动态 query，不使用 CGFE | 已完成 |
| A4 | Full DQ | `configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full.yml` | 加入 CCM + CGFE + 动态 query | 已完成 |

公共设置：

```yaml
dataset: VisDrone2019-DET
backbone: RT-DETRv2-R50VD
input_size: 640 x 640
total_batch_size: 4
epoches: 90
GPUs: 2,3
AMP: true
EMA: true
count_bins: [24, 42, 70]
dynamic_query_nums: [300, 500, 900, 1200]
```

## 指标说明

`test_coco_eval_bbox` 的 12 个值按如下顺序记录：

| 序号 | 指标 |
| --- | --- |
| 1 | AP |
| 2 | AP50 |
| 3 | AP75 |
| 4 | APs |
| 5 | APm |
| 6 | APl |
| 7 | AR1 |
| 8 | AR10 |
| 9 | AR100 |
| 10 | ARs |
| 11 | ARm |
| 12 | ARl |

## 汇总表

| 编号 | 状态 | Best Epoch | AP | AP50 | AP75 | APs | APm | APl | AR100 | Best 权重 | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A0 | 已完成 | 76 | 0.3211 | 0.5274 | 0.3251 | 0.2308 | 0.4315 | 0.6311 | 0.4849 | `rtdetrv2_pytorch/output/rtdetrv2_r50vd_6x_visdrone_p2/best.pth` | 原始 P2 对照基线 |
| A1 | 已完成 | 75 | 0.3196 | 0.5241 | 0.3243 | 0.2298 | 0.4309 | 0.6183 | 0.4838 | `rtdetrv2_pytorch/output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a1_count/best.pth` | final epoch 89 AP=0.3067；相对 A0 best AP -0.0014；冗余权重已清理 |
| A2 | 已完成 | 75 | 0.3190 | 0.5230 | 0.3236 | 0.2313 | 0.4304 | 0.6036 | 0.4858 | `rtdetrv2_pytorch/output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a2_cgfe/best.pth` | final epoch 89 AP=0.3062；2026-06-06 19:37 完成；冗余权重已清理 |
| A3 | 已完成 | 75 | 0.3203 | 0.5253 | 0.3261 | 0.2309 | 0.4315 | 0.6319 | 0.4866 | `rtdetrv2_pytorch/output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a3_dynamic_query/best.pth` | final epoch 89 AP=0.3058；2026-06-07 09:32 完成 |
| A4 | 已完成 | 79 | 0.3207 | 0.5269 | 0.3267 | 0.2345 | 0.4293 | 0.6090 | 0.4862 | `rtdetrv2_pytorch/output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full/best.pth` | final epoch 89 AP=0.3092；2026-06-07 23:28 完成；冗余权重已清理 |

## A0-A4 阶段总结

本轮实验的主线是验证 DQ-DETR 中的 CCM、CGFE 和 Dynamic Query 思想迁移到 RT-DETRv2-P2 后，对 VisDrone 小目标检测是否有效。所有实验均使用相同 COCO 预训练权重、相同 VisDrone 数据设置、相同训练轮数和相同 GPU 条件，因此主要对比各模块本身带来的影响。

按 best AP 排序：

| 排名 | 实验 | AP | 主要结论 |
| ---: | --- | ---: | --- |
| 1 | A0 P2 baseline | 0.3211 | 原始 P2 仍是整体 AP 最高的对照基线 |
| 2 | A4 Full DQ | 0.3207 | DQ 系列最高，整体 AP 基本追平 A0，小目标 APs 最优 |
| 3 | A3 Dynamic query | 0.3203 | 动态 query 最稳定，召回和定位质量略有收益 |
| 4 | A1 Count loss | 0.3196 | 单独加入计数辅助监督未带来整体提升 |
| 5 | A2 CGFE | 0.3190 | CGFE 对小目标/召回有轻微帮助，但大目标下降明显 |

相对 A0 best 的核心差值：

| 实验 | AP | AP50 | AP75 | APs | APm | APl | AR100 | 复盘判断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A1 | -0.0014 | -0.0033 | -0.0009 | -0.0010 | -0.0006 | -0.0128 | -0.0011 | Count loss 单独加入后收益不明显，可能主要起到辅助约束作用 |
| A2 | -0.0021 | -0.0043 | -0.0016 | +0.0005 | -0.0011 | -0.0275 | +0.0009 | CGFE 有一点小目标和召回收益，但对大目标干扰较强 |
| A3 | -0.0008 | -0.0021 | +0.0010 | +0.0001 | -0.0000 | +0.0008 | +0.0016 | Dynamic query 是当前最稳的单模块方向，整体接近 A0 |
| A4 | -0.0004 | -0.0004 | +0.0016 | +0.0037 | -0.0022 | -0.0221 | +0.0012 | Full DQ 小目标效果最好，但大目标下降限制了整体 AP |

模块贡献判断：

- **CCM + Count loss**：A1 没有超过 A0，说明计数辅助监督本身不足以直接提升检测精度，更适合作为 CGFE 或动态 query 的引导信号。
- **CGFE**：A2 的 APs 和 AR100 有轻微提升，但 APl 明显下降，说明 density-guided 特征增强可能更偏向密集小目标区域，对大目标特征表达存在干扰。
- **Dynamic query**：A3 的 AP 虽略低于 A0，但 AP75、APl、AR100 均有小幅提升，是当前迁移最稳的组件。
- **Full DQ**：A4 是 DQ 系列中最优结果，AP 几乎追平 A0，同时 APs 提升最大，说明完整方案对 VisDrone 小目标检测是有价值的；但 APl 下降较大，导致整体 AP 未能超过 baseline。

总体结论：

本轮实验没有证明 DQ-RTDETRv2-P2 在整体 AP 上显著超过原始 P2 baseline，但证明了完整 DQ 方案可以在几乎不损失整体 AP 的情况下提升小目标指标。A4 相比 A0 的 AP 仅低 `0.0004`，但 APs 提升 `0.0037`，AP75 提升 `0.0016`，AR100 提升 `0.0012`。因此，当前方案更适合作为“偏小目标/密集场景优化”的改进，而不是作为纯 mAP 提升版本直接定稿。

后续复盘建议：

- 优先做 A0 vs A4 的可视化对比，确认 APs 提升是否来自真实小目标召回提升。
- 对 A4 做 seed 复现实验，因为 A0 和 A4 的 AP 差距很小，可能存在随机波动。
- 优先尝试更温和的 dynamic query 数量，例如 `[300, 500, 700, 900]`，避免过多 query 带来误检或大目标干扰。
- 尝试削弱 CGFE 残差增强强度，或只在 P2/P3、小目标密集图像上启用 CGFE，以缓解 APl 下滑。
- A5/A6 若按指导文件原始定义继续跑，和当前 A2/A3 语义高度重合，优先级低于参数精调和可视化分析。

## A0 详细结果

输出目录：

```bash
rtdetrv2_pytorch/output/rtdetrv2_r50vd_6x_visdrone_p2
```

最佳 checkpoint：

```bash
rtdetrv2_pytorch/output/rtdetrv2_r50vd_6x_visdrone_p2/best.pth
```

最佳 epoch：76

| 指标 | 数值 |
| --- | ---: |
| AP | 0.321072 |
| AP50 | 0.527359 |
| AP75 | 0.325144 |
| APs | 0.230773 |
| APm | 0.431507 |
| APl | 0.631103 |
| AR1 | 0.123080 |
| AR10 | 0.366509 |
| AR100 | 0.484934 |
| ARs | 0.399852 |
| ARm | 0.604733 |
| ARl | 0.770096 |
| train_loss | 13.472278 |

最后 epoch：89

| 指标 | 数值 |
| --- | ---: |
| AP | 0.309302 |
| AP50 | 0.513653 |
| AP75 | 0.311659 |
| APs | 0.227261 |
| APm | 0.411014 |
| APl | 0.596597 |
| AR1 | 0.119172 |
| AR10 | 0.353346 |
| AR100 | 0.469145 |
| ARs | 0.390911 |
| ARm | 0.577492 |
| ARl | 0.723805 |
| train_loss | 12.600656 |

观察：

- A0 最佳 AP 出现在 epoch 76，后续训练到 epoch 89 时 AP 有回落。
- 后续消融对比建议统一使用各自 `best.pth` 对应的最佳 AP。
- A0 作为对照基线的核心指标为 AP 0.3211，AP50 0.5274，AP75 0.3251。

## 当前运行状态

更新时间：2026-06-08 11:25:49 CST

```bash
screen: 无运行中的 screen 会话
current experiment: 全部完成
latest progress: A4 completed at 2026-06-07 23:28:31, best epoch 79 AP=0.3207
GPU 2: 15 MiB, 0% util
GPU 3: 33 MiB, 0% util
```

A1 best：

| 指标 | 数值 |
| --- | ---: |
| best_epoch | 75 |
| AP | 0.319642 |
| AP50 | 0.524105 |
| AP75 | 0.324286 |
| APs | 0.229756 |
| APm | 0.430939 |
| APl | 0.618345 |
| AR100 | 0.483815 |

A1 screen 日志：

```bash
rtdetrv2_pytorch/output/rtdetrv2_p2_dq_a1_count/screen_train.log
```

A1 checkpoint 输出目录：

```bash
rtdetrv2_pytorch/output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a1_count
```

剩余队列脚本：

```bash
rtdetrv2_pytorch/run_dq_visdrone_a1_a4_ddp.sh
```

## 后续更新模板

每个实验完成后，在“汇总表”中补充如下字段：

```text
状态: 已完成
Best Epoch:
AP:
AP50:
AP75:
APs:
APm:
APl:
AR100:
Best 权重:
备注:
```

## A1 详细结果

实验内容：Count loss 消融。基于原始 P2 baseline，加入 CCM 计数分类分支和 `loss_count`，不启用 CGFE，不启用动态 query。

输出目录：

```bash
rtdetrv2_pytorch/output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a1_count
```

最佳 checkpoint：

```bash
rtdetrv2_pytorch/output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a1_count/best.pth
```

训练状态：已完成 90 epochs，screen 日志显示 `Training time 13:28:58`，完成时间 2026-06-04 23:57:22 CST。

最佳 epoch：75

| 指标 | 数值 |
| --- | ---: |
| AP | 0.319642 |
| AP50 | 0.524105 |
| AP75 | 0.324286 |
| APs | 0.229756 |
| APm | 0.430939 |
| APl | 0.618345 |
| AR1 | 0.122913 |
| AR10 | 0.363866 |
| AR100 | 0.483815 |
| ARs | 0.399742 |
| ARm | 0.602883 |
| ARl | 0.771536 |
| train_loss | 13.687840 |
| train_loss_count | 0.179109 |

最后 epoch：89

| 指标 | 数值 |
| --- | ---: |
| AP | 0.306706 |
| AP50 | 0.509110 |
| AP75 | 0.308875 |
| APs | 0.223038 |
| APm | 0.410683 |
| APl | 0.578878 |
| AR1 | 0.119300 |
| AR10 | 0.351218 |
| AR100 | 0.467056 |
| ARs | 0.389365 |
| ARm | 0.576196 |
| ARl | 0.733940 |
| train_loss | 12.703232 |
| train_loss_count | 0.164638 |

相对 A0 best：

| 指标 | A0 | A1 | 差值 |
| --- | ---: | ---: | ---: |
| AP | 0.321072 | 0.319642 | -0.001430 |
| AP50 | 0.527359 | 0.524105 | -0.003254 |
| AP75 | 0.325144 | 0.324286 | -0.000858 |
| APs | 0.230773 | 0.229756 | -0.001016 |
| APm | 0.431507 | 0.430939 | -0.000568 |
| APl | 0.631103 | 0.618345 | -0.012758 |
| AR100 | 0.484934 | 0.483815 | -0.001119 |

结论：A1 计数损失单独加入后与 A0 非常接近，但 best AP 未超过 A0，主要差距来自 AP50 和 APl。

A2 权重文件状态：已于 2026-06-06 清理 A2 冗余权重，删除 `checkpoint0000.pth` 到 `checkpoint0089.pth` 和 `last.pth`；当前保留 `best.pth`、`log.txt`、`eval/`、`summary/`，目录约 1.2G。

权重文件状态：已于 2026-06-05 清理 A1 冗余权重，删除 `checkpoint0000.pth` 到 `checkpoint0089.pth` 和 `last.pth`；当前保留 `best.pth`、`log.txt`、`screen_train.log`、`eval/`、`summary/`，目录约 1.2G。


## A3 详细结果

实验内容：Dynamic query 消融。基于 P2 baseline，加入 CCM 计数分类分支、`loss_count` 和动态 query，不启用 CGFE。

输出目录：

```bash
rtdetrv2_pytorch/output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a3_dynamic_query
```

最佳 checkpoint：

```bash
rtdetrv2_pytorch/output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a3_dynamic_query/best.pth
```

训练状态：已完成 90 epochs，完成时间 2026-06-07 09:32 CST。

最佳 epoch：75

| 指标 | 数值 |
| --- | ---: |
| AP | 0.320252 |
| AP50 | 0.525293 |
| AP75 | 0.326099 |
| APs | 0.230905 |
| APm | 0.431475 |
| APl | 0.631924 |
| AR1 | 0.123731 |
| AR10 | 0.366745 |
| AR100 | 0.486569 |
| ARs | 0.402183 |
| ARm | 0.607706 |
| ARl | 0.783903 |
| train_loss | 13.647965 |
| train_loss_count | 0.179332 |

最后 epoch：89，AP 0.305819。

相对 A0 best：

| 指标 | A0 | A3 | 差值 |
| --- | ---: | ---: | ---: |
| AP | 0.321072 | 0.320252 | -0.000820 |
| AP50 | 0.527359 | 0.525293 | -0.002066 |
| AP75 | 0.325144 | 0.326099 | +0.000955 |
| APs | 0.230773 | 0.230905 | +0.000132 |
| APm | 0.431507 | 0.431475 | -0.000032 |
| APl | 0.631103 | 0.631924 | +0.000821 |
| AR100 | 0.484934 | 0.486569 | +0.001635 |

结论：A3 的 best AP 略低于 A0，但 AP75、APs、APl 和 AR100 有小幅提升；动态 query 比 A1/A2 的整体 AP 更接近 A0。

## A4 详细结果

实验内容：Full DQ。基于 P2 baseline，同时启用 CCM、`loss_count`、CGFE 和动态 query。

输出目录：

```bash
rtdetrv2_pytorch/output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full
```

最佳 checkpoint：

```bash
rtdetrv2_pytorch/output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full/best.pth
```

训练状态：已完成 90 epochs，完成时间 2026-06-07 23:28:31 CST，screen 日志显示 `Training time 13:55:48`。

最佳 epoch：79

| 指标 | 数值 |
| --- | ---: |
| AP | 0.320686 |
| AP50 | 0.526945 |
| AP75 | 0.326740 |
| APs | 0.234521 |
| APm | 0.429318 |
| APl | 0.608966 |
| AR1 | 0.122810 |
| AR10 | 0.366857 |
| AR100 | 0.486182 |
| ARs | 0.402434 |
| ARm | 0.605564 |
| ARl | 0.764176 |
| train_loss | 13.395916 |
| train_loss_count | 0.174553 |

最后 epoch：89，AP 0.309222。

相对 A0 best：

| 指标 | A0 | A4 | 差值 |
| --- | ---: | ---: | ---: |
| AP | 0.321072 | 0.320686 | -0.000386 |
| AP50 | 0.527359 | 0.526945 | -0.000414 |
| AP75 | 0.325144 | 0.326740 | +0.001596 |
| APs | 0.230773 | 0.234521 | +0.003748 |
| APm | 0.431507 | 0.429318 | -0.002189 |
| APl | 0.631103 | 0.608966 | -0.022137 |
| AR100 | 0.484934 | 0.486182 | +0.001248 |

结论：A4 的整体 AP 略低于 A0，但优于 A1/A2/A3；主要收益体现在 AP75、APs 和 AR100，小目标表现是当前所有实验中最高。

A4 权重文件状态：已于 2026-06-08 清理 A4 冗余权重，删除 `checkpoint0000.pth` 到 `checkpoint0089.pth` 和 `last.pth`；当前保留 `best.pth`、`log.txt`、`eval/`、`summary/`，目录约 1.2G。

## 常用检查命令

查看后台任务：

```bash
screen -ls
```

查看 GPU：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
```

查看当前 screen 日志：

```bash
tail -n 80 -f /home/wang.kui/proj/RTDETR/DQ_RTDETRv2_P2/rtdetrv2_pytorch/output/rtdetrv2_p2_dq_a4_full/screen_train.log
```

解析某个实验的 best 和 final 指标：

```bash
cd /home/wang.kui/proj/RTDETR/DQ_RTDETRv2_P2/rtdetrv2_pytorch
python3 - <<'PY'
import json
from pathlib import Path

log_path = Path('output/rtdetrv2_r50vd_6x_visdrone_p2/log.txt')
labels = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]

for name, row in [('best', max(rows, key=lambda r: r['test_coco_eval_bbox'][0])), ('last', rows[-1])]:
    print(name, 'epoch', row['epoch'])
    for label, value in zip(labels, row['test_coco_eval_bbox']):
        print(f'{label}: {value:.6f}')
    print(f"train_loss: {row['train_loss']:.6f}")
PY
```
