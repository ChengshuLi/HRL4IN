#!/bin/bash

gpu="0"
reward_type="sparse"
pos="random"
lr="1e-3"
meta_lr="1e-5"  # 1e-4, 1e-5
num_steps="1024"
ec="0.03"
time_scale="4"
name="exp"
run="0"

log_dir="hrl_reward_"$reward_type"_pos_"$pos"_lr_"$lr"_meta_lr_"$meta_lr"_nsteps_"$num_steps"_ec_"$ec"_ts_"$time_scale"_"$name"_run_"$run
echo $log_dir

python -u train_hrl_toy_env.py \
   --use-gae \
   --sim-gpu-id $gpu \
   --pth-gpu-id $gpu \
   --lr $lr \
   --meta-lr $meta_lr \
   --clip-param 0.1 \
   --value-loss-coef 0.5 \
   --num-train-processes 32 \
   --num-eval-processes 1 \
   --num-steps $num_steps \
   --num-mini-batch 32 \
   --num-updates 50000 \
   --use-linear-lr-decay \
   --use-linear-clip-decay \
   --entropy-coef $ec \
   --log-interval 1 \
   --experiment-folder "ckpt/"$log_dir \
   --time-scale $time_scale \
   --meta-agent-normalize-advantage \
   --use-action-masks \
   --meta-gamma 0.999 \
   --checkpoint-interval 100 \
   --checkpoint-index -1 \
   --env-type "toy" \
   --config-file "map.yaml" \
   --num-eval-episodes 100 \
   --eval-only
