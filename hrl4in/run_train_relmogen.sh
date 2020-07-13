#!/bin/bash

gpu_g="0"
gpu_c="1"
irs="30.0"
sgr="0.0"
lr="1e-4"
meta_lr="1e-5"        # 1e-4, 1e-5
fr_lr="0"             # 0, 100
init_std_dev_xy="0.6" # 0.6, 1.2
init_std_dev_z="0.1"
failed_pnt="0.0"      # 0.0, -0.2
num_steps="256"
ext_col="0.0"         # 0.0, 0.5, 1.0, 2.0
arena="push_door"
run="0"

log_dir="/result/hrl4in_baseline_"$arena"_"$run
echo $log_dir
mkdir -p $log_dir

nohup python -u train_hrl_relmogen.py \
   --use-gae \
   --sim-gpu-id $gpu_g \
   --pth-gpu-id $gpu_c \
   --lr $lr \
   --meta-lr $meta_lr \
   --freeze-lr-n-updates $fr_lr \
   --clip-param 0.1 \
   --value-loss-coef 0.5 \
   --num-train-processes 9 \
   --num-eval-processes 1 \
   --num-steps $num_steps \
   --num-mini-batch 1 \
   --num-updates 100000 \
   --use-linear-lr-decay \
   --use-linear-clip-decay \
   --entropy-coef 0.01 \
   --log-interval 5 \
   --experiment-folder $log_dir \
   --time-scale 30 \
   --intrinsic-reward-scaling $irs \
   --subgoal-achieved-reward $sgr \
   --subgoal-init-std-dev $init_std_dev_xy $init_std_dev_xy $init_std_dev_z \
   --subgoal-failed-penalty $failed_pnt \
   --use-action-masks \
   --meta-agent-normalize-advantage \
   --extrinsic-collision-reward-weight $ext_col \
   --meta-gamma 0.99 \
   --checkpoint-interval 50 \
   --checkpoint-index -1 \
   --env-type "relmogen" \
   --config-file "fetch_interactive_nav_s2r_mp_continuous.yaml" \
   --model-ids Avonia,Avonia,Avonia,candcenter,candcenter,candcenter,gates_jan20,gates_jan20,gates_jan20 \
   --arena $arena > $log_dir/log &
   # --env-mode "headless" \
   # --num-eval-episodes 100 \
   # --eval-only

