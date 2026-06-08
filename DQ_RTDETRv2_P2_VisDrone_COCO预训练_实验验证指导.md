# DQ-RTDETRv2-P2：基于 DQ-DETR 思想的 VisDrone 有效性验证指导文件

> 目标：在 `hanqiuguang8-stack/RT-DETRv2-P2` 的 P2 版 RT-DETRv2 基础上，参考 `hoiliu-0801/DQ-DETR` 的 CCM / CGFE / Dynamic Query 思想，实现一个适合 VisDrone 小目标检测的轻量改进版，并通过消融实验验证有效性。  
> 训练方式：使用 COCO 预训练权重初始化，在 VisDrone2019-DET 数据集上训练。  
> 说明：COCO 预训练权重路径暂时留空，后续由你手动指定。

---

## 1. 本文件适用仓库

主工程仓库：

```bash
https://github.com/hanqiuguang8-stack/RT-DETRv2-P2
```

参考思想仓库：

```bash
https://github.com/hoiliu-0801/DQ-DETR
```

主工程中需要重点参考 / 修改的文件：

```text
rtdetrv2_pytorch/src/zoo/rtdetr/rtdetr.py
rtdetrv2_pytorch/src/zoo/rtdetr/hybrid_encoder_P2.py
rtdetrv2_pytorch/src/zoo/rtdetr/rtdetr_decoder.py
rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_decoder.py
rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_criterion.py
rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml
```

DQ-DETR 中重点参考的文件：

```text
models/dqdetr/ccm.py
models/dqdetr/cgfe.py
models/dqdetr/deformable_transformer.py
models/dqdetr/dqdetr.py
```

---

## 2. 改进方案总览

原始 RT-DETRv2-P2 流程可以理解为：

```text
Input Image
   ↓
Backbone
   ↓
S2 / S3 / S4 / S5
   ↓
HybridEncoderP2
   ├─ S5 → AIFI → F5
   └─ S2/S3/S4/F5 → CCFF → Fused Features
                              ↓
                         Flatten
                              ↓
                    Query Selection
                              ↓
                         Decoder + Head
                              ↓
                         Detection Result
```

本实验方案改为：

```text
Input Image
   ↓
Backbone
   ↓
S2 / S3 / S4 / S5
   ↓
HybridEncoderP2
   ├─ S5 → AIFI → F5
   └─ S2/S3/S4/F5 → CCFF → Fused Features
                              ↓
S2 → CCM → Density Map ─────→ CGFE
        ↓                     ↓
   Count Level          Position-aware Features
        ↓                     ↓
        └────────→ Count-guided Query Selection
                              ↓
                          Decoder + Head
                              ↓
                         Detection Result
```

核心思路：

```text
S2 负责提供小目标密度和数量先验；
CCFF 负责正常多尺度融合；
Density Map 负责增强融合特征的位置感知能力；
Count Level 负责指导 Query Selection 阶段的候选 Query 数量。
```

---

## 3. 为什么这样设计

### 3.1 S2 作为 CCM 输入

S2 是 stride=4 的高分辨率特征，保留了更多小目标细节。VisDrone 中的 pedestrian、people、bicycle、motorcycle 等类别尺寸较小，过深层的 S4/S5 容易丢失位置细节。因此用 S2 做计数分支输入是合理的。

### 3.2 Density Map 作用在融合后的特征上

不要把 CGFE 放在 CCFF 之前。否则增强后的特征可能会被后续 FPN / PAN / CCFF 重新融合后稀释。

推荐顺序：

```text
S2/S3/S4/F5 → CCFF → Fused Features
Fused Features + Density Map → CGFE → Position-aware Features
```

### 3.3 Count Level 指导 Query Selection

数量等级不直接参与分类或框回归，而是用于决定 query 数量：

```text
Low     → 较少 queries
Medium  → 中等 queries
High    → 较多 queries
```

示例：

```python
dynamic_query_nums = [500, 900, 1200]
```

也可以使用 4 档：

```python
dynamic_query_nums = [300, 500, 900, 1200]
```

建议先用 3 档或 4 档，具体阈值应根据 VisDrone 训练集目标数量统计确定。

---

## 4. 实验目录建议

在主工程中新增或修改以下文件：

```text
rtdetrv2_pytorch/
├── configs/
│   └── rtdetrv2/
│       ├── rtdetrv2_r50vd_6x_visdrone_p2.yml
│       ├── rtdetrv2_r50vd_6x_visdrone_p2_dq_a1_count.yml
│       ├── rtdetrv2_r50vd_6x_visdrone_p2_dq_a2_cgfe.yml
│       ├── rtdetrv2_r50vd_6x_visdrone_p2_dq_a3_dynamic_query.yml
│       └── rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full.yml
│
├── tools/
│   ├── analyze_visdrone_count_bins.py
│   ├── check_dq_forward.py
│   └── export_dq_experiment_table.py
│
└── src/
    └── zoo/
        └── rtdetr/
            ├── dq_modules.py                  # 新增
            ├── hybrid_encoder_P2.py            # 修改
            ├── rtdetr.py                       # 修改
            ├── rtdetr_decoder.py               # 按实际配置引用决定是否修改
            ├── rtdetrv2_decoder.py             # 按实际配置引用决定是否修改
            └── rtdetrv2_criterion.py           # 修改
```

注意：

```text
如果配置文件中使用的是 RTDETRTransformerv2，则主要修改 rtdetrv2_decoder.py；
如果配置文件中使用的是 RTDETRTransformer，则主要修改 rtdetr_decoder.py。
```

Codex 执行时应先查看当前 YAML 中实际注册的 decoder 名称，再决定修改哪个文件。

---

## 5. 第一步：统计 VisDrone 每张图的目标数量

### 5.1 目的

Count Level 的阈值不能完全照搬 DQ-DETR，因为 DQ-DETR 主要针对 AI-TOD。VisDrone 的图像尺度、目标密度和类别分布不同，应该先统计训练集中每张图的目标数量，再设置 bins。

### 5.2 新建脚本

文件：

```text
rtdetrv2_pytorch/tools/analyze_visdrone_count_bins.py
```

参考代码：

```python
import json
import argparse
from collections import Counter
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann",
        type=str,
        required=True,
        help="COCO 格式 VisDrone train annotation json 路径"
    )
    args = parser.parse_args()

    with open(args.ann, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_ids = [img["id"] for img in coco["images"]]
    count_dict = {img_id: 0 for img_id in img_ids}

    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        img_id = ann["image_id"]
        if img_id in count_dict:
            count_dict[img_id] += 1

    counts = np.array(list(count_dict.values()), dtype=np.int64)

    print("num_images:", len(counts))
    print("min:", int(counts.min()))
    print("max:", int(counts.max()))
    print("mean:", float(counts.mean()))
    print("std:", float(counts.std()))
    print("p25:", float(np.percentile(counts, 25)))
    print("p50:", float(np.percentile(counts, 50)))
    print("p75:", float(np.percentile(counts, 75)))
    print("p90:", float(np.percentile(counts, 90)))
    print("p95:", float(np.percentile(counts, 95)))

    bins_3 = [
        int(np.percentile(counts, 33)),
        int(np.percentile(counts, 66)),
    ]

    bins_4 = [
        int(np.percentile(counts, 25)),
        int(np.percentile(counts, 50)),
        int(np.percentile(counts, 75)),
    ]

    print("suggested count_bins_3:", bins_3)
    print("suggested count_bins_4:", bins_4)

    hist = Counter(counts.tolist())
    print("top count frequencies:")
    for k, v in hist.most_common(20):
        print(k, v)


if __name__ == "__main__":
    main()
```

运行示例：

```bash
cd rtdetrv2_pytorch

python tools/analyze_visdrone_count_bins.py \
  --ann dataset/VisDrone2019/annotations/instances_train.json
```

如果你的 annotation 路径不同，请改成实际路径。

---

## 6. 第二步：新增 dq_modules.py

文件：

```text
rtdetrv2_pytorch/src/zoo/rtdetr/dq_modules.py
```

参考代码：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalCountingModule(nn.Module):
    """
    输入:
        p2_feat: [B, C, H, W]

    输出:
        count_logits: [B, num_count_classes]
        density_map:  [B, 1, H, W]
        density_feat: [B, C, H, W]

    说明:
        参考 DQ-DETR 中 CategoricalCounting 的思想，但改为适合 RT-DETRv2-P2 的轻量版本。
    """

    def __init__(
        self,
        in_channels=256,
        hidden_channels=256,
        num_count_classes=4,
        use_dilated=True,
    ):
        super().__init__()

        dilation = 2 if use_dilated else 1
        padding = dilation

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )

        self.density_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, 1, kernel_size=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(hidden_channels, num_count_classes)

    def forward(self, p2_feat):
        density_feat = self.stem(p2_feat)
        density_map = self.density_head(density_feat)

        pooled = self.pool(density_feat).flatten(1)
        count_logits = self.cls_head(pooled)

        return count_logits, density_map, density_feat


class DensityGuidedCGFE(nn.Module):
    """
    输入:
        feats: list[Tensor]，CCFF/PAN 输出后的融合特征，多尺度 [F2, F3, F4, F5]
        density_map: [B, 1, H2, W2]

    输出:
        enhanced_feats: list[Tensor]，增强后的多尺度特征

    核心公式:
        F_i_out = F_i + alpha_i * sigmoid(resize(DensityMap)) * F_i

    说明:
        采用残差式增强，避免直接替换原特征导致训练不稳定。
    """

    def __init__(self, num_levels=4, init_alpha=0.5):
        super().__init__()
        self.num_levels = num_levels
        self.alpha = nn.Parameter(torch.ones(num_levels) * init_alpha)

    def forward(self, feats, density_map):
        if density_map is None:
            return feats

        enhanced = []

        for i, feat in enumerate(feats):
            attn = F.interpolate(
                density_map,
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            attn = torch.sigmoid(attn)

            scale = torch.sigmoid(self.alpha[i])
            out = feat + scale * attn * feat
            enhanced.append(out)

        return enhanced


def build_count_targets(targets, count_bins, device):
    """
    根据每张图的 GT 目标数量生成数量等级标签。

    参数:
        targets: list[dict]，每个 dict 至少包含 labels
        count_bins:
            3 档时，例如 [40, 120]，输出类别 0/1/2
            4 档时，例如 [20, 80, 200]，输出类别 0/1/2/3
    """
    labels = []

    for t in targets:
        n = len(t["labels"])

        level = 0
        for b in count_bins:
            if n >= b:
                level += 1
            else:
                break

        labels.append(level)

    return torch.tensor(labels, dtype=torch.long, device=device)
```

---

## 7. 第三步：修改 HybridEncoderP2

文件：

```text
rtdetrv2_pytorch/src/zoo/rtdetr/hybrid_encoder_P2.py
```

### 7.1 修改目标

在 `HybridEncoderP2` 中完成：

```text
1. 输入 S2/S3/S4/S5；
2. 取投影后的 S2 作为 CCM 输入；
3. S2/S3/S4/S5 仍正常经过 AIFI + FPN/PAN/CCFF；
4. CCFF 输出的融合特征与 Density Map 进入 CGFE；
5. 返回增强后的融合特征，同时把 count_logits / density_map / count_level 信息传给 decoder 或顶层模型。
```

### 7.2 推荐做法

由于原始 `RTDETR.forward()` 是：

```python
x = self.backbone(x)
x = self.encoder(x)
x = self.decoder(x, targets)
return x
```

因此建议让 encoder 返回一个 tuple：

```python
enhanced_feats, dq_info = self.encoder(x)
```

然后修改 `rtdetr.py`，把 `dq_info` 传入 decoder：

```python
out = self.decoder(enhanced_feats, targets, dq_info=dq_info)
```

### 7.3 在 `HybridEncoderP2.__init__()` 中新增参数

找到：

```python
class HybridEncoderP2(nn.Module):
```

在 `__init__()` 参数中新增：

```python
use_dq=False,
num_count_classes=4,
count_bins=[20, 80, 200],
use_cgfe=True,
```

然后新增导入：

```python
from .dq_modules import CategoricalCountingModule, DensityGuidedCGFE
```

并在 `__init__()` 中加入：

```python
self.use_dq = use_dq
self.use_cgfe = use_cgfe
self.count_bins = count_bins
self.num_count_classes = num_count_classes

if self.use_dq:
    self.ccm = CategoricalCountingModule(
        in_channels=hidden_dim,
        hidden_channels=hidden_dim,
        num_count_classes=num_count_classes,
    )

    self.cgfe = DensityGuidedCGFE(
        num_levels=len(in_channels),
        init_alpha=0.5,
    )
else:
    self.ccm = None
    self.cgfe = None
```

### 7.4 在 `forward()` 中插入 CCM 和 CGFE

在 `forward()` 中，原始逻辑大致是：

```python
proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

# encoder
# top-down fpn
# bottom-up pan
return outs
```

改成：

```python
def forward(self, feats):
    assert len(feats) == len(self.in_channels)

    proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

    dq_info = None

    if self.use_dq:
        # S2 是第 0 层，shape: [B, 256, H/4, W/4]
        p2_feat_for_count = proj_feats[0]
        count_logits, density_map, density_feat = self.ccm(p2_feat_for_count)

        dq_info = {
            "count_logits": count_logits,
            "density_map": density_map,
        }

    # 原始 AIFI / encoder 流程保持不变
    if self.num_encoder_layers > 0:
        for i, enc_ind in enumerate(self.use_encoder_idx):
            h, w = proj_feats[enc_ind].shape[2:]
            src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)

            if self.training or self.eval_spatial_size is None:
                pos_embed = self.build_2d_sincos_position_embedding(
                    w, h, self.hidden_dim, self.pe_temperature
                ).to(src_flatten.device)
            else:
                pos_embed = getattr(self, f"pos_embed{enc_ind}", None).to(src_flatten.device)

            memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
            proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(
                -1, self.hidden_dim, h, w
            ).contiguous()

    # 原始 FPN/PAN/CCFF 流程保持不变
    inner_outs = [proj_feats[-1]]

    for idx in range(len(self.in_channels) - 1, 0, -1):
        feat_heigh = inner_outs[0]
        feat_low = proj_feats[idx - 1]

        feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
        inner_outs[0] = feat_heigh

        upsample_feat = F.interpolate(feat_heigh, scale_factor=2.0, mode="nearest")
        inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
            torch.concat([upsample_feat, feat_low], dim=1)
        )
        inner_outs.insert(0, inner_out)

    outs = [inner_outs[0]]

    for idx in range(len(self.in_channels) - 1):
        feat_low = outs[-1]
        feat_height = inner_outs[idx + 1]

        downsample_feat = self.downsample_convs[idx](feat_low)
        out = self.pan_blocks[idx](
            torch.concat([downsample_feat, feat_height], dim=1)
        )
        outs.append(out)

    # 关键：CGFE 放在 FPN/PAN/CCFF 之后
    if self.use_dq and self.use_cgfe:
        outs = self.cgfe(outs, density_map)

    if self.use_dq:
        return outs, dq_info

    return outs
```

---

## 8. 第四步：修改 rtdetr.py

文件：

```text
rtdetrv2_pytorch/src/zoo/rtdetr/rtdetr.py
```

原始代码：

```python
def forward(self, x, targets=None):
    x = self.backbone(x)
    x = self.encoder(x)
    x = self.decoder(x, targets)
    return x
```

改成：

```python
def forward(self, x, targets=None):
    x = self.backbone(x)

    enc_out = self.encoder(x)

    if isinstance(enc_out, tuple):
        feats, dq_info = enc_out
    else:
        feats, dq_info = enc_out, None

    out = self.decoder(feats, targets, dq_info=dq_info)
    return out
```

如果 baseline decoder 暂时还没有 `dq_info` 参数，可以先使用兼容写法：

```python
try:
    out = self.decoder(feats, targets, dq_info=dq_info)
except TypeError:
    out = self.decoder(feats, targets)
    if dq_info is not None and isinstance(out, dict):
        out.update(dq_info)
return out
```

正式实验时推荐修改 decoder，使其显式支持 `dq_info`。

---

## 9. 第五步：修改 Decoder，实现 Count-guided Query Selection

实际修改文件取决于 YAML 中 decoder 的注册名：

```text
优先检查 configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml

如果使用 RTDETRTransformerv2：
    修改 rtdetrv2_decoder.py

如果使用 RTDETRTransformer：
    修改 rtdetr_decoder.py
```

下面以 `rtdetr_decoder.py` 风格为例说明。

### 9.1 在 decoder `__init__()` 中新增参数

```python
use_dq=False,
use_dynamic_query=False,
count_bins=[20, 80, 200],
dynamic_query_nums=[300, 500, 900, 1200],
```

并设置：

```python
self.use_dq = use_dq
self.use_dynamic_query = use_dynamic_query
self.count_bins = count_bins
self.dynamic_query_nums = dynamic_query_nums
```

### 9.2 新增函数：由 count_logits 计算动态 query 数

```python
def _get_dynamic_query_num(self, dq_info):
    """
    为了保证 batch 内 tensor shape 一致，取 batch 内最大 Count Level 对应的 query 数量。
    """
    if (not self.use_dq) or (not self.use_dynamic_query) or dq_info is None:
        return self.num_queries

    count_logits = dq_info.get("count_logits", None)
    if count_logits is None:
        return self.num_queries

    count_level = count_logits.detach().argmax(dim=1)
    max_level = int(count_level.max().item())

    max_level = min(max_level, len(self.dynamic_query_nums) - 1)
    return int(self.dynamic_query_nums[max_level])
```

### 9.3 修改 `_get_decoder_input()`

原始代码中有类似逻辑：

```python
_, topk_ind = torch.topk(
    enc_outputs_class.max(-1).values,
    self.num_queries,
    dim=1
)
```

改为：

```python
def _get_decoder_input(
    self,
    memory,
    spatial_shapes,
    denoising_class=None,
    denoising_bbox_unact=None,
    query_num=None,
):
    bs, _, _ = memory.shape

    if query_num is None:
        query_num = self.num_queries

    query_num = min(int(query_num), memory.shape[1])

    if self.training or self.eval_spatial_size is None:
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
    else:
        anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

    memory = valid_mask.to(memory.dtype) * memory

    output_memory = self.enc_output(memory)
    enc_outputs_class = self.enc_score_head(output_memory)
    enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

    _, topk_ind = torch.topk(
        enc_outputs_class.max(-1).values,
        query_num,
        dim=1,
    )

    reference_points_unact = enc_outputs_coord_unact.gather(
        dim=1,
        index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]),
    )
    enc_topk_bboxes = F.sigmoid(reference_points_unact)

    if denoising_bbox_unact is not None:
        reference_points_unact = torch.concat(
            [denoising_bbox_unact, reference_points_unact],
            dim=1,
        )

    enc_topk_logits = enc_outputs_class.gather(
        dim=1,
        index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]),
    )

    if self.learnt_init_query:
        # 注意：如果使用 learnt_init_query，需要 tgt_embed 数量至少 >= max(dynamic_query_nums)
        target = self.tgt_embed.weight[:query_num].unsqueeze(0).tile([bs, 1, 1])
    else:
        target = output_memory.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]),
        )

    target = target.detach()

    if denoising_class is not None:
        target = torch.concat([denoising_class, target], dim=1)

    return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits
```

### 9.4 修改 decoder `forward()`

原始：

```python
def forward(self, feats, targets=None):
```

改成：

```python
def forward(self, feats, targets=None, dq_info=None):
```

在 `_get_encoder_input()` 后加入：

```python
memory, spatial_shapes, level_start_index = self._get_encoder_input(feats)

query_num = self._get_dynamic_query_num(dq_info)
```

然后调用 `_get_decoder_input()`：

```python
target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = self._get_decoder_input(
    memory,
    spatial_shapes,
    denoising_class,
    denoising_bbox_unact,
    query_num=query_num,
)
```

输出 dict 中加入：

```python
if dq_info is not None:
    out.update(dq_info)
    out["dynamic_query_num"] = torch.as_tensor(
        query_num,
        device=out["pred_logits"].device,
        dtype=torch.long,
    )
```

### 9.5 Denoising 训练注意事项

如果打开 `num_denoising > 0`，原始 denoising 逻辑中可能使用 `self.num_queries` 构造 attention mask。动态 query 数量改变后，attention mask 的形状可能不匹配。

建议按阶段处理：

```text
阶段 1：先只做 A1 / A2，固定 query 数，不改 denoising。
阶段 2：做 A3 / A4 时，先设置 num_denoising=0 跑通 forward 和训练。
阶段 3：再把 denoising 中的 self.num_queries 替换为 query_num。
```

也可以在调用 denoising 函数时把 `query_num` 传进去：

```python
get_contrastive_denoising_training_group(
    targets,
    self.num_classes,
    query_num,
    self.denoising_class_embed,
    ...
)
```

但要同步检查 `dn_meta['dn_num_split']` 是否与 decoder 输出维度匹配。

---

## 10. 第六步：修改 Criterion，加入 count loss

文件：

```text
rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_criterion.py
```

### 10.1 在 `__init__()` 中新增参数

```python
use_count_loss=False,
count_bins=[20, 80, 200],
lambda_count=0.2,
```

设置：

```python
self.use_count_loss = use_count_loss
self.count_bins = count_bins
self.lambda_count = lambda_count
```

### 10.2 新增 loss 函数

在文件顶部加入：

```python
from .dq_modules import build_count_targets
```

在类中新增：

```python
def loss_count(self, outputs, targets):
    if "count_logits" not in outputs:
        device = next(iter(outputs.values())).device
        return {"loss_count": torch.as_tensor(0.0, device=device)}

    count_logits = outputs["count_logits"]
    count_targets = build_count_targets(
        targets,
        self.count_bins,
        device=count_logits.device,
    )

    loss = F.cross_entropy(count_logits, count_targets)
    return {"loss_count": loss * self.lambda_count}
```

### 10.3 在 `forward()` 最后加入

在 `return losses` 之前加入：

```python
if self.use_count_loss:
    losses.update(self.loss_count(outputs, targets))
```

注意：

```text
loss_count 不参与 Hungarian matching。
它是图像级辅助损失，只监督 CCM 的 Count Level 预测。
```

---

## 11. 第七步：配置文件设置

### 11.1 Baseline 配置

保留原文件不动：

```text
configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml
```

训练 baseline 时通过 `-t` 加载 COCO 预训练权重：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml \
  -t /path/to/coco_pretrained_weight.pth \
  --use-amp
```

其中：

```text
/path/to/coco_pretrained_weight.pth
```

后续由你指定。

---

### 11.2 新建 Full 配置

新建：

```text
configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full.yml
```

建议继承 baseline：

```yaml
__include__:
  - ./rtdetrv2_r50vd_6x_visdrone_p2.yml

output_dir: ./output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full

HybridEncoderP2:
  use_dq: True
  use_cgfe: True
  num_count_classes: 4
  count_bins: [20, 80, 200]

RTDETRTransformerv2:
  use_dq: True
  use_dynamic_query: True
  count_bins: [20, 80, 200]
  dynamic_query_nums: [300, 500, 900, 1200]

RTDETRCriterionv2:
  use_count_loss: True
  count_bins: [20, 80, 200]
  lambda_count: 0.2
```

如果实际配置中 decoder 名称不是 `RTDETRTransformerv2`，Codex 需要替换成实际注册名。

---

## 12. 第八步：消融实验设计

必须保证所有实验使用相同条件：

```text
数据集：VisDrone2019-DET
初始化：同一个 COCO 预训练权重
输入分辨率：与 baseline 保持一致
batch size：与 baseline 保持一致
训练 epoch：建议 90 epoch 或与原配置一致
优化器、学习率、数据增强：保持一致
随机种子：固定
```

### 12.1 实验表

| 实验编号 | 模型 | CCM | Density Map | CGFE | Dynamic Query | Count Loss | 目的 |
|---|---|---|---|---|---|---|---|
| A0 | RT-DETRv2-P2 | 否 | 否 | 否 | 否 | 否 | baseline |
| A1 | + CCM | 是 | 是 | 否 | 否 | 是 | 验证计数辅助监督是否有用 |
| A2 | + CCM + CGFE | 是 | 是 | 是 | 否 | 是 | 验证 Density Map 引导特征增强是否有效 |
| A3 | + CCM + Dynamic Query | 是 | 是 | 否 | 是 | 是 | 验证数量等级指导 Query 是否有效 |
| A4 | Full | 是 | 是 | 是 | 是 | 是 | 验证完整方案 |
| A5 | Full without Count Level | 是 | 是 | 是 | 否 | 是 | 验证 Query 动态选择贡献 |
| A6 | Full without CGFE | 是 | 是 | 否 | 是 | 是 | 验证 CGFE 贡献 |

推荐至少完成：

```text
A0、A1、A2、A3、A4
```

如果时间不够，至少完成：

```text
A0、A2、A4
```

---

## 13. 训练命令

### 13.1 Baseline

```bash
cd rtdetrv2_pytorch

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml \
  -t /path/to/coco_pretrained_weight.pth \
  --use-amp \
  --seed=0
```

### 13.2 A1：只加 CCM + Count Loss

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a1_count.yml \
  -t /path/to/coco_pretrained_weight.pth \
  --use-amp \
  --seed=0
```

### 13.3 A2：CCM + CGFE，固定 Query 数

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a2_cgfe.yml \
  -t /path/to/coco_pretrained_weight.pth \
  --use-amp \
  --seed=0
```

### 13.4 A3：CCM + Dynamic Query，不加 CGFE

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a3_dynamic_query.yml \
  -t /path/to/coco_pretrained_weight.pth \
  --use-amp \
  --seed=0
```

### 13.5 A4：完整方案

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full.yml \
  -t /path/to/coco_pretrained_weight.pth \
  --use-amp \
  --seed=0
```

---

## 14. 测试命令

每组实验训练完成后，使用 best checkpoint 测试：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full.yml \
  -r output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full/best.pth \
  --test-only
```

Baseline 测试：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml \
  -r output/rtdetrv2_r50vd_6x_visdrone_p2/best.pth \
  --test-only
```

---

## 15. Forward 检查脚本

新增：

```text
rtdetrv2_pytorch/tools/check_dq_forward.py
```

参考代码：

```python
import torch


def check_output(out):
    print("output keys:", out.keys())

    assert "pred_logits" in out
    assert "pred_boxes" in out

    print("pred_logits:", out["pred_logits"].shape)
    print("pred_boxes:", out["pred_boxes"].shape)

    if "count_logits" in out:
        print("count_logits:", out["count_logits"].shape)

    if "density_map" in out:
        print("density_map:", out["density_map"].shape)

    if "dynamic_query_num" in out:
        print("dynamic_query_num:", out["dynamic_query_num"])


@torch.no_grad()
def main():
    """
    该脚本需要 Codex 根据工程实际 build_model 方式补全。
    目标不是直接训练，而是先验证一次 forward 能否跑通。
    """

    # 示例伪代码：
    # from src.core import YAMLConfig
    # cfg = YAMLConfig("configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full.yml")
    # model = cfg.model.cuda().eval()

    model = None
    assert model is not None, "请根据当前工程实际配置方式补全 model build 代码"

    x = torch.randn(2, 3, 640, 640).cuda()
    out = model(x, targets=None)

    check_output(out)


if __name__ == "__main__":
    main()
```

Codex 任务：

```text
根据 RT-DETRv2-P2 工程实际的配置加载方式，补全 check_dq_forward.py 中的 model build 逻辑。
```

---

## 16. 有效性验证指标

### 16.1 主指标

记录：

```text
AP
AP50
AP75
APs
APm
APl
AR1
AR10
AR100
ARs
ARm
ARl
```

如果评估工具输出 per-class AP，也记录 VisDrone 10 类：

```text
pedestrian
people
bicycle
car
van
truck
tricycle
awning-tricycle
bus
motor
```

注意：类别名称以你当前 COCO 转换脚本中的 category 名称为准。

---

### 16.2 本方案特别关注指标

重点看：

```text
APs 是否提升
ARs 是否提升
pedestrian / people / bicycle / motorcycle 是否提升
密集场景图片的召回是否提升
推理速度是否明显下降
显存是否明显增加
```

如果 A4 比 A0：

```text
APs 提升 ≥ 0.5
ARs 提升 ≥ 0.5
整体 AP 不下降或小幅提升
FPS 下降不超过 10%
```

可以认为方案初步有效。

---

## 17. 日志记录表

每组实验完成后填写：

| 实验 | 配置文件 | 预训练权重 | best epoch | AP | AP50 | AP75 | APs | ARs | FPS | 显存 | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A0 | p2 baseline | COCO |  |  |  |  |  |  |  |  |  |
| A1 | + CCM | COCO |  |  |  |  |  |  |  |  |  |
| A2 | + CCM + CGFE | COCO |  |  |  |  |  |  |  |  |  |
| A3 | + Dynamic Query | COCO |  |  |  |  |  |  |  |  |  |
| A4 | Full | COCO |  |  |  |  |  |  |  |  |  |

---

## 18. 结果分析模板

实验完成后，按照以下模板写结论：

```text
与 RT-DETRv2-P2 baseline 相比，加入 CCM 后，模型获得了图像级目标数量监督。
如果 A1 的 APs 或 ARs 提升，说明数量辅助监督有助于小目标特征学习。

在 A2 中，Density Map 被用于增强 CCFF 输出的融合特征。
如果 A2 的 APs 和小目标类别 AP 提升，说明密度图提供的位置先验有效。

在 A3 中，Count Level 被用于动态调整 Query Selection 的候选数量。
如果 A3 的 ARs 提升，说明密集场景中更多 query 有助于召回更多小目标。

A4 综合使用 CCM、CGFE 和 Dynamic Query。
如果 A4 的 APs/ARs 最优，且整体 AP 没有明显下降，则说明 DQ-DETR 思想迁移到 RT-DETRv2-P2 是有效的。
```

---

## 19. 可视化验证

除了数值指标，还建议做 3 类可视化。

### 19.1 Density Map 可视化

保存 `density_map`：

```python
density = torch.sigmoid(outputs["density_map"])
```

把 density resize 到原图大小，叠加在原图上，观察热点是否集中在车辆、人群、小目标密集区域。

### 19.2 Query 点分布可视化

把 Query Selection 得到的参考点可视化到原图上。

观察：

```text
A0 的 query 是否比较分散；
A4 的 query 是否更多集中到小目标密集区域。
```

### 19.3 检测结果对比

选 20 张图片：

```text
10 张小目标密集图
5 张稀疏图
5 张中等目标数量图
```

对比：

```text
A0 baseline
A2 + CGFE
A4 full
```

观察漏检是否减少。

---

## 20. 可能出现的问题与解决方案

### 20.1 训练时报 forward 参数错误

错误类似：

```text
forward() got an unexpected keyword argument 'dq_info'
```

原因：

```text
decoder 还没有改成 def forward(self, feats, targets=None, dq_info=None)
```

解决：

```text
修改当前 YAML 实际调用的 decoder 文件。
```

---

### 20.2 输出 loss 中没有 loss_count

原因可能是：

```text
criterion 没有开启 use_count_loss；
outputs 中没有 count_logits；
rtdetr.py 没有把 dq_info 传到 decoder 输出 dict。
```

检查：

```python
print(outputs.keys())
```

必须看到：

```text
count_logits
```

---

### 20.3 Dynamic Query 后维度不匹配

常见原因：

```text
denoising training 的 attention mask 仍然按 self.num_queries 构造；
但实际 topK query_num 变了。
```

解决优先级：

```text
1. debug 阶段设置 num_denoising=0；
2. 跑通后再把 denoising 中的 self.num_queries 替换为 query_num；
3. 检查 dn_meta['dn_num_split'] 是否和 decoder 输出维度一致。
```

---

### 20.4 COCO 预训练权重加载时 shape mismatch

VisDrone 类别数通常是 10，而 COCO 是 80。分类头 shape 不一致属于正常现象。

解决原则：

```text
backbone、encoder、decoder 大部分权重加载；
classification head 不匹配的权重跳过或重新初始化。
```

如果工程的 `-t` 微调逻辑已经自动处理不匹配，直接使用：

```bash
-t /path/to/coco_pretrained_weight.pth
```

如果报错，需要 Codex 修改权重加载逻辑：遇到 shape 不一致时跳过。

参考代码：

```python
def load_pretrained_ignore_mismatch(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    model_state = model.state_dict()
    new_state = {}

    skipped = []

    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            new_state[k] = v
        else:
            skipped.append(k)

    model_state.update(new_state)
    model.load_state_dict(model_state, strict=True)

    print(f"Loaded params: {len(new_state)}")
    print(f"Skipped params: {len(skipped)}")
    for k in skipped[:50]:
        print("skip:", k)
```

---

## 21. 给 Codex 的总任务提示词

可以直接复制给 Codex：

```text
你现在在 RT-DETRv2-P2 工程中工作，目标是参考 DQ-DETR 的 CCM、CGFE、Dynamic Query 思想，实现 DQ-RTDETRv2-P2 并通过 VisDrone 消融实验验证有效性。

主仓库：
https://github.com/hanqiuguang8-stack/RT-DETRv2-P2

参考仓库：
https://github.com/hoiliu-0801/DQ-DETR

请重点参考：
- RT-DETRv2-P2:
  - rtdetrv2_pytorch/src/zoo/rtdetr/rtdetr.py
  - rtdetrv2_pytorch/src/zoo/rtdetr/hybrid_encoder_P2.py
  - rtdetrv2_pytorch/src/zoo/rtdetr/rtdetr_decoder.py
  - rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_decoder.py
  - rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_criterion.py
  - rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml
- DQ-DETR:
  - models/dqdetr/ccm.py
  - models/dqdetr/cgfe.py
  - models/dqdetr/deformable_transformer.py
  - models/dqdetr/dqdetr.py

实现要求：
1. 新增 rtdetrv2_pytorch/src/zoo/rtdetr/dq_modules.py。
2. 在 dq_modules.py 中实现：
   - CategoricalCountingModule
   - DensityGuidedCGFE
   - build_count_targets
3. 修改 HybridEncoderP2：
   - 取投影后的 S2 作为 CCM 输入；
   - 输出 count_logits 和 density_map；
   - 保持 S2/S3/S4/S5 正常经过 AIFI + FPN/PAN/CCFF；
   - 在 CCFF 输出之后，用 density_map 通过 CGFE 增强融合特征；
   - 返回 enhanced_feats 和 dq_info。
4. 修改 rtdetr.py：
   - 支持 encoder 返回 tuple；
   - 把 dq_info 传入 decoder。
5. 修改当前配置实际使用的 decoder：
   - forward 增加 dq_info 参数；
   - 根据 count_logits 得到 count level；
   - 用 count level 决定动态 query_num；
   - topk selection 从固定 self.num_queries 改为 query_num；
   - 输出 dict 中加入 count_logits、density_map、dynamic_query_num。
6. 修改 rtdetrv2_criterion.py：
   - 增加 loss_count；
   - 根据 targets 中每张图 GT 数量生成 count target；
   - count loss 用 cross entropy；
   - 默认 lambda_count=0.2。
7. 新增 VisDrone 目标数量统计脚本：
   - tools/analyze_visdrone_count_bins.py
8. 新增 forward 检查脚本：
   - tools/check_dq_forward.py
9. 新增配置文件：
   - rtdetrv2_r50vd_6x_visdrone_p2_dq_a1_count.yml
   - rtdetrv2_r50vd_6x_visdrone_p2_dq_a2_cgfe.yml
   - rtdetrv2_r50vd_6x_visdrone_p2_dq_a3_dynamic_query.yml
   - rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full.yml
10. 所有实验都使用同一个 COCO 预训练权重，通过 -t 参数加载。预训练权重路径我后续手动指定。
11. 先保证单张 forward 能跑通，再进行正式训练。
12. 对比 A0/A1/A2/A3/A4 的 AP、AP50、AP75、APs、ARs、小目标类别 AP、FPS、显存。
13. 不要修改原始 baseline 配置文件，所有新实验使用新配置文件。
```

---

## 22. 最终验收标准

Codex 完成后，至少应满足：

```text
1. baseline 配置仍可正常训练；
2. 新配置 A1/A2/A3/A4 能被 YAML 正确构建；
3. check_dq_forward.py 能跑通；
4. outputs 中包含：
   - pred_logits
   - pred_boxes
   - count_logits
   - density_map
   - dynamic_query_num
5. 训练 loss 中包含：
   - loss_count
6. A0/A1/A2/A3/A4 至少完成 test-only 评估；
7. 形成实验结果表；
8. 能明确回答：
   - CCM 是否有效？
   - Density Map 引导 CGFE 是否有效？
   - Count Level 指导 Dynamic Query 是否有效？
   - 完整方案是否优于 RT-DETRv2-P2 baseline？
```

---

## 23. 推荐最终论文表述

可以在论文或报告中这样描述：

```text
本文在 RT-DETRv2-P2 的基础上引入计数引导的动态查询机制。首先，利用 S2 高分辨率特征构建 Categorical Counting Module，以获得目标密度图和图像级数量等级。随后，保持原始 S2-S5 多尺度特征正常经过 HybridEncoderP2 中的 AIFI 和 CCFF 模块，得到融合后的多尺度特征。为了避免计数引导信息被后续多尺度融合稀释，本文将 Counting-Guided Feature Enhancement 放置在 CCFF 之后，利用 Density Map 对融合特征进行残差式位置增强，生成带有目标密度先验的 position-aware features。最后，将 Count Level 引入 Query Selection 阶段，根据图像中目标数量动态调整候选 query 数量，使密集小目标场景获得更多查询，从而提升小目标检测召回能力。
```

---

## 24. 推荐实验结论判断

如果实验结果如下：

```text
A1 相比 A0：loss_count 收敛，APs/ARs 小幅提升
A2 相比 A1：APs 提升，说明 Density Map 位置增强有效
A3 相比 A1：ARs 提升，说明 Dynamic Query 对召回有效
A4 相比 A0：APs、ARs、部分小目标类别 AP 提升，整体 AP 不明显下降
```

则可以得出：

```text
DQ-DETR 的计数引导和动态查询思想可以有效迁移到 RT-DETRv2-P2，并对 VisDrone 小目标检测具有正向作用。
```

如果出现：

```text
A4 APs 提升但整体 AP 下降
```

则说明：

```text
方案对小目标更敏感，但可能引入背景误检，需要调整 lambda_count、density enhancement alpha 或 dynamic query bins。
```

如果出现：

```text
A2/A4 没提升
```

优先检查：

```text
1. Density Map 是否真的聚焦目标区域；
2. Count bins 是否不合理；
3. CGFE 是否增强过强；
4. Count loss 权重是否太大；
5. Dynamic query 数量是否过多导致误检增加。
```
