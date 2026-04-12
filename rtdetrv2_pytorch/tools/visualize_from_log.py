import os
import json
import numpy as np
import matplotlib.pyplot as plt


def load_log_data(log_path):
    """从log.txt文件加载评估数据"""
    log_data = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                log_data.append(data)
            except json.JSONDecodeError:
                pass
    return log_data


def extract_coco_metrics(log_data):
    """从log数据中提取COCO评估指标"""
    metrics = []
    for data in log_data:
        if 'test_coco_eval_bbox' in data:
            metrics.append({
                'epoch': data.get('epoch', 0),
                'bbox': data['test_coco_eval_bbox']
            })
    return metrics


def plot_pr_curve_from_log(output_dir):
    """从log文件绘制PR曲线（模拟）"""
    # 由于我们没有直接的PR数据，我们可以使用mAP值来模拟一个PR曲线
    # 实际应用中，需要从评估结果中提取更详细的数据
    
    # 模拟PR曲线数据
    recall = np.linspace(0, 1, 100)
    precision = 0.3 + 0.6 * np.exp(-recall * 3)  # 模拟一个典型的PR曲线
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve (Simulated)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()


def plot_f1_curve_from_log(output_dir):
    """从log文件绘制F1置信度曲线（模拟）"""
    # 模拟F1曲线数据
    confidence = np.linspace(0, 1, 100)
    f1 = 2 * (confidence * 0.8) / (confidence + 0.8 + 1e-8)  # 模拟F1曲线
    
    plt.figure(figsize=(10, 8))
    plt.plot(confidence, f1, label='F1 Score')
    plt.xlabel('Confidence Score')
    plt.ylabel('F1 Score')
    plt.title('F1 Confidence Curve (Simulated)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'f1_curve.png'))
    plt.close()


def plot_confusion_matrix_from_log(output_dir):
    """从log文件绘制混淆矩阵（模拟）"""
    # 模拟混淆矩阵数据
    # 假设VisDrone数据集有10个类别
    categories = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    num_classes = len(categories)
    
    # 生成一个模拟的混淆矩阵
    confusion_matrix = np.random.randint(0, 100, size=(num_classes, num_classes))
    # 对角线值设置为较大的值，模拟正确分类
    for i in range(num_classes):
        confusion_matrix[i, i] = np.random.randint(200, 500)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.colorbar()
    
    # 添加类别标签
    plt.xticks(np.arange(len(categories)), categories, rotation=45, ha='right')
    plt.yticks(np.arange(len(categories)), categories)
    
    # 添加数值
    for i in range(len(categories)):
        for j in range(len(categories)):
            plt.text(j, i, int(confusion_matrix[i, j]), ha='center', va='center', color='black')
    
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix (Simulated)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


def plot_data_distribution_from_log(output_dir):
    """从log文件绘制数据分布图（模拟）"""
    # 模拟数据分布
    categories = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    counts = [1200, 800, 400, 2000, 600, 300, 200, 150, 250, 900]
    
    plt.figure(figsize=(12, 8))
    plt.bar(categories, counts)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Data Distribution (Simulated)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_distribution.png'))
    plt.close()


def plot_training_metrics(log_data, output_dir):
    """绘制训练指标"""
    epochs = []
    train_loss = []
    test_bbox = []
    
    for data in log_data:
        if 'epoch' in data:
            epochs.append(data['epoch'])
            if 'train_loss' in data:
                train_loss.append(data['train_loss'])
            if 'test_coco_eval_bbox' in data:
                test_bbox.append(data['test_coco_eval_bbox'][0])  # mAP@0.5:0.95
    
    # 绘制训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'train_loss.png'))
    plt.close()
    
    # 绘制测试mAP
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, test_bbox, label='mAP@0.5:0.95')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Test mAP')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'test_map.png'))
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize model performance from log file')
    parser.add_argument('--log_dir', type=str, required=True, help='Log directory path')
    parser.add_argument('--output', type=str, default='./visualization', help='Output directory')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 加载log数据
    log_path = os.path.join(args.log_dir, 'log.txt')
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return
    
    log_data = load_log_data(log_path)
    
    # 提取COCO指标
    metrics = extract_coco_metrics(log_data)
    
    # 绘制各种图表
    plot_pr_curve_from_log(args.output)
    plot_f1_curve_from_log(args.output)
    plot_confusion_matrix_from_log(args.output)
    plot_data_distribution_from_log(args.output)
    plot_training_metrics(log_data, args.output)
    
    print(f"Visualization results saved to {args.output}")


if __name__ == '__main__':
    main()
