"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime
import numpy as np

import torch 

from ..misc import dist_utils, profiler_utils

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
        
        # ---------------- 增加：提取并打印每个类别的 AP 指标 ----------------
        try:
            if dist_utils.is_main_process() and coco_evaluator is not None:
                coco_eval = coco_evaluator.coco_eval.get('bbox')
                if coco_eval is not None:
                    # 获取底层的精度评估矩阵
                    precisions = coco_eval.eval['precision']
                    catIds = coco_eval.params.catIds
                    cats = coco_eval.cocoGt.loadCats(catIds)

                    print("\n" + "="*65)
                    print(f"{'Category':<20} | {'mAP@0.5:0.95':<18} | {'mAP@0.5':<18}")
                    print("-" * 65)

                    for i, cat in enumerate(cats):
                        # precisions 矩阵维度: [T(IoU), R(Recall), K(类别), A(面积), M(最大检测数)]
                        # T=:, R=:, K=i, A=0(所有面积), M=-1(当前配置的最大检测数)
                        
                        # 计算当前类别的 mAP@0.5:0.95
                        p_map = precisions[:, :, i, 0, -1]
                        p_map = p_map[p_map > -1] # -1 代表该类别在GT中不存在，需过滤
                        ap_map = np.mean(p_map) if len(p_map) > 0 else 0.0

                        # 计算当前类别的 AP50 (T=0 代表 IoU=0.5)
                        p_50 = precisions[0, :, i, 0, -1]
                        p_50 = p_50[p_50 > -1]
                        ap_50 = np.mean(p_50) if len(p_50) > 0 else 0.0

                        print(f"{cat['name']:<20} | {ap_map:<18.3f} | {ap_50:<18.3f}")
                    print("="*65 + "\n")
        except Exception as e:
            if dist_utils.is_main_process():
                print(f"\n[Warning] 无法提取单类别指标: {e}")
        # --------------------------------------------------------------
        
        return