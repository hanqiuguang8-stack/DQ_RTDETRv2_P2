# DQ_RTDETRv2_P2

本仓库是在 RT-DETRv2-P2 基础上进行的 VisDrone 小目标检测实验工程。实验参考 DQ-DETR 的计数引导思想，在 RT-DETRv2-P2 中加入 CCM、CGFE 和 Dynamic Query，用于验证这些模块迁移到 VisDrone 密集小目标场景后的效果。

## 实验目标

- 以原始 RT-DETRv2-R50VD + P2 作为 baseline。
- 引入 Categorical Counting Module，预测图像级目标数量等级和 density map。
- 使用 density map 引导 CGFE，对融合特征进行位置增强。
- 使用 count level 指导 query selection，在密集场景中动态调整 query 数量。
- 通过 A0-A4 消融实验评估各模块贡献。

## 主要实验设置

| 项目 | 设置 |
| --- | --- |
| 数据集 | VisDrone2019-DET |
| 模型 | RT-DETRv2-R50VD + P2 |
| 输入尺寸 | 640 x 640 |
| 训练轮数 | 90 epochs |
| 总 batch size | 4 |
| 基础 query 数 | 900 |
| Dynamic query 档位 | `[300, 500, 900, 1200]` |
| Count bins | `[24, 42, 70]` |
| 预训练 | COCO 预训练权重微调 |

## 消融实验

| 编号 | 实验 | 主要改动 |
| --- | --- | --- |
| A0 | P2 baseline | 原始 RT-DETRv2-R50VD + P2 |
| A1 | Count loss | 加入 CCM 和 count loss，不启用 CGFE，不启用动态 query |
| A2 | CGFE | 加入 CCM + CGFE，固定 query |
| A3 | Dynamic query | 加入 CCM + 动态 query，不启用 CGFE |
| A4 | Full DQ | 同时启用 CCM、CGFE 和动态 query |

## 实验结果

以下为各实验 best checkpoint 的 COCO bbox 指标：

| 实验 | Best Epoch | AP | AP50 | AP75 | APs | APm | APl | AR100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 76 | 0.3211 | 0.5274 | 0.3251 | 0.2308 | 0.4315 | 0.6311 | 0.4849 |
| A1 | 75 | 0.3196 | 0.5241 | 0.3243 | 0.2298 | 0.4309 | 0.6183 | 0.4838 |
| A2 | 75 | 0.3190 | 0.5230 | 0.3236 | 0.2313 | 0.4304 | 0.6036 | 0.4858 |
| A3 | 75 | 0.3203 | 0.5253 | 0.3261 | 0.2309 | 0.4315 | 0.6319 | 0.4866 |
| A4 | 79 | 0.3207 | 0.5269 | 0.3267 | 0.2345 | 0.4293 | 0.6090 | 0.4862 |

## 阶段结论

A0 仍是整体 AP 最高的 baseline，AP 为 `0.3211`。A4 是 DQ 系列中最好的结果，AP 为 `0.3207`，与 A0 只差 `0.0004`。

A4 的主要收益体现在小目标和召回上：相对 A0，APs 提升 `0.0037`，AP75 提升 `0.0016`，AR100 提升 `0.0012`。但 A4 的 APl 下降 `0.0221`，说明当前 CGFE + Dynamic Query 组合更偏向小目标密集区域，对大目标有一定干扰。

因此，本轮实验没有证明 DQ-RTDETRv2-P2 能显著超过原始 P2 baseline 的整体 AP，但说明完整 DQ 方案对 VisDrone 小目标检测有正向价值，适合作为小目标/密集场景优化方向继续调参。

## 关键文件

| 文件 | 说明 |
| --- | --- |
| `DQ_RTDETRv2_P2_VisDrone_COCO预训练_实验验证指导.md` | 实验实现与验证指导 |
| `DQ_RTDETRv2_P2_VisDrone_COCO预训练_实验结果.md` | A0-A4 实验结果与复盘总结 |
| `rtdetrv2_pytorch/src/zoo/rtdetr/dq_modules.py` | CCM 与 CGFE 模块实现 |
| `rtdetrv2_pytorch/configs/rtdetrv2/*dq*.yml` | A1-A4 消融实验配置 |
| `rtdetrv2_pytorch/run_dq_visdrone_*.sh` | 后台训练脚本 |

## 说明

模型权重文件未上传到 GitHub，仓库中仅保留代码、配置、实验文档和日志。复现实验时需要自行准备 VisDrone 数据集和 COCO 预训练权重。
