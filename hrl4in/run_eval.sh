#!/bin/bash

python -u train_hrl_gibson.py \
   --use-gae \
   --sim-gpu-id 1 \
   --pth-gpu-id 1 \
   --lr 1e-4 \
   --meta-lr 1e-5 \
   --freeze-lr-n-updates 0 \
   --clip-param 0.1 \
   --value-loss-coef 0.5 \
   --num-train-processes 1 \
   --num-eval-processes 1 \
   --num-steps 1024 \
   --num-mini-batch 1 \
   --num-updates 50000 \
   --use-linear-lr-decay \
   --use-linear-clip-decay \
   --use-action-masks \
   --entropy-coef 0.01 \
   --log-interval 5 \
   --experiment-folder "icra_eval/hrl_reward_dense_pos_fixed_sgm_arm_world_irs_30.0_sgr_0.0_lr_1e-4_meta_lr_1e-5_fr_lr_0_death_30.0_init_std_0.6_0.6_0.1_failed_pnt_0.0_nsteps_1024_ext_col_0.0_6x6_from_scr_minus_death_stage_reward_run_3" \
   --time-scale 50 \
   --intrinsic-reward-scaling 30.0 \
   --subgoal-achieved-reward 0.0 \
   --subgoal-init-std-dev 0.6 0.6 0.1 \
   --subgoal-failed-penalty 0.0 \
   --meta-gamma 0.99 \
   --use-action-masks \
   --meta-agent-normalize-advantage \
   --checkpoint-interval 100 \
   --checkpoint-index 3500 \
   --env-type "interactive_gibson" \
   --config-file "jr_interactive_nav.yaml" \
   --eval-only \
   --env-mode "headless" \
   --arena "complex_hl_ll" \
   --num-eval-episodes 100

