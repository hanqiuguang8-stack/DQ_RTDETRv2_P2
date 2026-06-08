"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime
import numpy as np
from pathlib import Path

import torch

from ..misc import dist_utils 

from ..misc import profiler_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        
        for epoch in range(start_epcoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, 
                self.criterion, 
                self.postprocessor, 
                self.val_dataloader, 
                self.evaluator, 
                self.device
            )

            # TODO 
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)
            
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat['epoch'] == epoch and self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)
                
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        # ---------------- 增加：提取打印每个类别的指标，并绘制单独的 PR 曲线 ----------------
        try:
            if dist_utils.is_main_process() and coco_evaluator is not None:
                coco_eval = coco_evaluator.coco_eval.get('bbox')
                if coco_eval is not None:
                    # 获取底层数据
                    precisions = coco_eval.eval['precision']
                    catIds = coco_eval.params.catIds
                    cats = coco_eval.cocoGt.loadCats(catIds)
                    
                    # 生成用于绘图的 Recall 点 (COCO 默认 101 个点)
                    recalls = np.linspace(0, 1, 101)

                    print("\n" + "="*65)
                    print(f"{'Category':<20} | {'mAP@0.5:0.95':<18} | {'mAP@0.5':<18}")
                    print("-" * 65)

                    # 创建图表保存目录
                    # 兼容 python 的 pathlib
                    save_dir = Path(self.cfg.output_dir) if isinstance(self.cfg.output_dir, str) else self.output_dir
                    plot_dir = save_dir / "evaluation_plots"
                    plot_dir.mkdir(parents=True, exist_ok=True)
                    
                    import matplotlib.pyplot as plt

                    for i, cat in enumerate(cats):
                        cat_name = cat['name']
                        
                        # --- 1. 计算控制台打印的 mAP 数值 ---
                        p_map = precisions[:, :, i, 0, -1]
                        p_map_valid = p_map[p_map > -1] 
                        ap_map = np.mean(p_map_valid) if len(p_map_valid) > 0 else 0.0

                        p_50 = precisions[0, :, i, 0, -1] # T=0 (IoU=0.5)
                        p_50_valid = p_50[p_50 > -1]
                        ap_50 = np.mean(p_50_valid) if len(p_50_valid) > 0 else 0.0

                        print(f"{cat_name:<20} | {ap_map:<18.3f} | {ap_50:<18.3f}")
                        
                        # --- 2. 绘制并保存当前类别的独立 PR 曲线 ---
                        plt.figure(figsize=(8, 6))
                        
                        # p_50 就是 IoU=0.5 时的 Precision 数组，长度与 recalls 对应
                        valid_mask = p_50 > -1
                        if valid_mask.any():
                            plt.plot(recalls[valid_mask], p_50[valid_mask], 
                                     label=f'{cat_name} (AP@0.5={ap_50:.3f})', 
                                     color='#1f77b4', linewidth=2)
                        
                        plt.title(f'PR Curve: {cat_name.upper()} @ IoU=0.50', fontsize=16)
                        plt.xlabel('Recall', fontsize=14)
                        plt.ylabel('Precision', fontsize=14)
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.grid(True, linestyle='--', alpha=0.6)
                        plt.legend(loc='lower left', fontsize=12)
                        plt.tight_layout()
                        
                        # 保存图像，文件名包含类别名
                        file_path = plot_dir / f"PR_Curve_IoU50_{cat_name.replace(' ', '_')}.png"
                        plt.savefig(file_path, dpi=300)
                        plt.close()

                    print("="*65 + "\n")
                    print(f"[Info] 所有类别的独立 PR 曲线已保存至: {plot_dir}")
                    
        except Exception as e:
            if dist_utils.is_main_process():
                print(f"\n[Warning] 无法提取单类别指标或绘图: {e}")
        # --------------------------------------------------------------
        
        return