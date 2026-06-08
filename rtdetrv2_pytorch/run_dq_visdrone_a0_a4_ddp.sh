#!/usr/bin/env bash
set -euo pipefail

cd /home/wang.kui/proj/RTDETR/DQ_RTDETRv2_P2/rtdetrv2_pytorch
export CUDA_VISIBLE_DEVICES=2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WEIGHT='/home/wang.kui/proj/RTDETR/RT-DETR_copy copy/rtdetrv2_r50vd_6x_coco_ema.pth'
MASTER_PORT=29623

echo "DQ RTDETRv2-P2 VisDrone queue started at $(date)"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

mkdir -p output/rtdetrv2_r50vd_6x_visdrone_p2
echo "[A0] start $(date)" | tee -a output/rtdetrv2_r50vd_6x_visdrone_p2/screen_train.log
conda run --no-capture-output -n RTDETR torchrun --master_port=${MASTER_PORT} --nproc_per_node=2 tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2.yml -t "${WEIGHT}" --use-amp --seed=0 >> output/rtdetrv2_r50vd_6x_visdrone_p2/screen_train.log 2>&1
echo "[A0] done $(date)" | tee -a output/rtdetrv2_r50vd_6x_visdrone_p2/screen_train.log

mkdir -p output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a1_count
echo "[A1] start $(date)" | tee -a output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a1_count/screen_train.log
conda run --no-capture-output -n RTDETR torchrun --master_port=${MASTER_PORT} --nproc_per_node=2 tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a1_count.yml -t "${WEIGHT}" --use-amp --seed=0 >> output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a1_count/screen_train.log 2>&1
echo "[A1] done $(date)" | tee -a output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a1_count/screen_train.log

mkdir -p output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a2_cgfe
echo "[A2] start $(date)" | tee -a output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a2_cgfe/screen_train.log
conda run --no-capture-output -n RTDETR torchrun --master_port=${MASTER_PORT} --nproc_per_node=2 tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a2_cgfe.yml -t "${WEIGHT}" --use-amp --seed=0 >> output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a2_cgfe/screen_train.log 2>&1
echo "[A2] done $(date)" | tee -a output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a2_cgfe/screen_train.log

mkdir -p output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a3_dynamic_query
echo "[A3] start $(date)" | tee -a output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a3_dynamic_query/screen_train.log
conda run --no-capture-output -n RTDETR torchrun --master_port=${MASTER_PORT} --nproc_per_node=2 tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a3_dynamic_query.yml -t "${WEIGHT}" --use-amp --seed=0 >> output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a3_dynamic_query/screen_train.log 2>&1
echo "[A3] done $(date)" | tee -a output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a3_dynamic_query/screen_train.log

mkdir -p output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full
echo "[A4] start $(date)" | tee -a output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full/screen_train.log
conda run --no-capture-output -n RTDETR torchrun --master_port=${MASTER_PORT} --nproc_per_node=2 tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full.yml -t "${WEIGHT}" --use-amp --seed=0 >> output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full/screen_train.log 2>&1
echo "[A4] done $(date)" | tee -a output/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full/screen_train.log

echo "DQ RTDETRv2-P2 VisDrone queue finished at $(date)"
