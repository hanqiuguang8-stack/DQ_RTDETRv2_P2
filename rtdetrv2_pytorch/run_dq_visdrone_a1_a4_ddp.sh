#!/usr/bin/env bash
set -euo pipefail
cd "/home/wang.kui/proj/RTDETR/DQ_RTDETRv2_P2/rtdetrv2_pytorch"
export CUDA_VISIBLE_DEVICES=2,3
export OMP_NUM_THREADS=1
MASTER_PORT=${MASTER_PORT:-29624}
WEIGHT="/home/wang.kui/proj/RTDETR/RT-DETR_copy copy/rtdetrv2_r50vd_6x_coco_ema.pth"
echo "[$(date +'%F %T')] resume DQ RTDETRv2 P2 VisDrone A1-A4"
mkdir -p "output/rtdetrv2_p2_dq_a1_count"
echo "[$(date +'%F %T')] START A1_COUNT cfg=configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a1_count.yml" | tee -a "output/rtdetrv2_p2_dq_a1_count/screen_train.log"
conda run --no-capture-output -n RTDETR torchrun --master_port=${MASTER_PORT} --nproc_per_node=2 tools/train.py -c "configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a1_count.yml" -t "${WEIGHT}" --use-amp --seed=0 >> "output/rtdetrv2_p2_dq_a1_count/screen_train.log" 2>&1
echo "[$(date +'%F %T')] DONE A1_COUNT" | tee -a "output/rtdetrv2_p2_dq_a1_count/screen_train.log"
mkdir -p "output/rtdetrv2_p2_dq_a2_cgfe"
echo "[$(date +'%F %T')] START A2_CGFE cfg=configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a2_cgfe.yml" | tee -a "output/rtdetrv2_p2_dq_a2_cgfe/screen_train.log"
conda run --no-capture-output -n RTDETR torchrun --master_port=${MASTER_PORT} --nproc_per_node=2 tools/train.py -c "configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a2_cgfe.yml" -t "${WEIGHT}" --use-amp --seed=0 >> "output/rtdetrv2_p2_dq_a2_cgfe/screen_train.log" 2>&1
echo "[$(date +'%F %T')] DONE A2_CGFE" | tee -a "output/rtdetrv2_p2_dq_a2_cgfe/screen_train.log"
mkdir -p "output/rtdetrv2_p2_dq_a3_dynamic_query"
echo "[$(date +'%F %T')] START A3_DYNAMIC_QUERY cfg=configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a3_dynamic_query.yml" | tee -a "output/rtdetrv2_p2_dq_a3_dynamic_query/screen_train.log"
conda run --no-capture-output -n RTDETR torchrun --master_port=${MASTER_PORT} --nproc_per_node=2 tools/train.py -c "configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a3_dynamic_query.yml" -t "${WEIGHT}" --use-amp --seed=0 >> "output/rtdetrv2_p2_dq_a3_dynamic_query/screen_train.log" 2>&1
echo "[$(date +'%F %T')] DONE A3_DYNAMIC_QUERY" | tee -a "output/rtdetrv2_p2_dq_a3_dynamic_query/screen_train.log"
mkdir -p "output/rtdetrv2_p2_dq_a4_full"
echo "[$(date +'%F %T')] START A4_FULL cfg=configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full.yml" | tee -a "output/rtdetrv2_p2_dq_a4_full/screen_train.log"
conda run --no-capture-output -n RTDETR torchrun --master_port=${MASTER_PORT} --nproc_per_node=2 tools/train.py -c "configs/rtdetrv2/rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full.yml" -t "${WEIGHT}" --use-amp --seed=0 >> "output/rtdetrv2_p2_dq_a4_full/screen_train.log" 2>&1
echo "[$(date +'%F %T')] DONE A4_FULL" | tee -a "output/rtdetrv2_p2_dq_a4_full/screen_train.log"
echo "[$(date +'%F %T')] all resumed experiments finished"
