# RT-DETRv2-P2

RT-DETRv2-P2 是一个基于 RT-DETRv2 的目标检测模型，特别添加了 P2 检测头，用于提高小目标检测性能。

## 项目结构

```
RT-DETR/
├── benchmark/         # 基准测试目录
├── hubconf.py         # PyTorch Hub 配置文件
└── rtdetrv2_pytorch/  # 主要代码目录
    ├── configs/       # 配置文件
    ├── dataset/       # 数据集
    ├── tools/         # 工具脚本
    ├── src/           # 源代码
    └── requirements.txt # 依赖文件
```

## 环境配置

1. 克隆仓库

```bash
git clone https://github.com/hanqiuguang8-stack/RT-DETRv2-P2.git
cd RT-DETRv2-P2
```

2. 安装依赖

```bash
cd rtdetrv2_pytorch
pip install -r requirements.txt
```

3. 安装额外依赖（如果需要）

```bash
pip install tensorboard faster-coco-eval
```

## 数据集准备

本项目使用 VisDrone 数据集进行训练和评估。请按照以下步骤准备数据集：

1. 下载 VisDrone 数据集：https://github.com/VisDrone/VisDrone-Dataset
2. 解压数据集到 `rtdetrv2_pytorch/dataset` 目录
3. 确保数据集结构如下：

```
dataset/
└── VisDrone2019-DET-train/
    ├── images/
    └── annotations/
└── VisDrone2019-DET-val/
    ├── images/
    └── annotations/
└── VisDrone2019-DET-test-dev/
    ├── images/
    └── annotations/
```

## 训练

### 基础训练

使用以下命令开始训练：

```bash
# 使用单个GPU训练
python tools/train.py \
    -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml

# 使用指定GPU训练
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml
```

### 从 checkpoint 恢复训练

```bash
python tools/train.py \
    -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml \
    -r path/to/checkpoint.pth
```

### 微调训练

```bash
python tools/train.py \
    -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml \
    -t path/to/checkpoint.pth
```

## 推理

### 模型评估

使用以下命令评估模型性能：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml \
    -r path/to/best.pth \
    --test-only
```

### 导出模型

#### 导出为 ONNX 格式

```bash
python tools/export_onnx.py \
    -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml \
    -r path/to/best.pth \
    -o output/model.onnx
```

#### 导出为 TensorRT 格式

```bash
python tools/export_trt.py \
    -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml \
    -r path/to/best.pth \
    -o output/model.trt
```

## 配置文件说明

主要配置文件 `configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml` 包含以下关键参数：

- **模型架构**：使用 HybridEncoderP2 和 RTDETRTransformerv2
- **特征层**：包含 P2 (Stride 4)、P3、P4、P5 四个特征层
- **训练参数**：batch size、学习率、优化器等
- **输出目录**：模型权重和日志的保存位置

## 可视化

### 从日志生成可视化

```bash
python tools/visualize_from_log.py \
    --log-file path/to/log.txt \
    --output-dir output/visualization
```

### 生成 PR 曲线和其他评估指标

```bash
python tools/generate_pr_curve_from_eval.py
```

## 模型性能

在 VisDrone 数据集上的评估结果：

| 类别 | mAP@0.5:0.95 | mAP@0.5 |
|------|-------------|---------|
| pedestrian | 0.303 | 0.607 |
| people | 0.227 | 0.532 |
| bicycle | 0.149 | 0.310 |
| car | 0.629 | 0.872 |
| van | 0.396 | 0.554 |
| truck | 0.315 | 0.461 |
| tricycle | 0.237 | 0.410 |
| motorcycle | 0.139 | 0.231 |
| large-vehicle | 0.491 | 0.651 |
| others | 0.308 | 0.626 |

## 注意事项

1. **大文件处理**：模型权重文件较大，建议使用 Git LFS 管理
2. **GPU 内存**：训练时建议使用至少 16GB GPU 内存
3. **数据集路径**：确保数据集路径正确配置在配置文件中
4. **评估指标**：使用 COCO 评估协议计算 mAP 等指标

## 引用

如果您使用本项目，请引用以下论文：

```
@article{lyu2023rtdetr,
  title={RT-DETR: DETR with Vision Transformer Backbone and Rotated Bounding Box},
  author={Lyu, Wenyu and Zhang, Shangliang and Li, Chao and Li, Guanzhong and Sun, Jian and Wang, Yuning},
  journal={arXiv preprint arXiv:2304.08069},
  year={2023}
}
```

## 联系方式

如果您有任何问题或建议，请通过 GitHub Issues 与我们联系。
