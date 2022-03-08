import os
import argparse
from typing import Dict

#--------------------------------------------------------
# Wrapper around habitat_baselines/run.py 
# Encapsulates experiment definitions and common overrides.
#--------------------------------------------------------

# List experiments for quick multi-launch from commandline.
#NOTE: See the examples/template below for formatting
experiments: Dict[str, Dict[str,str]] = {
    # "template":{
    #     "description": "",
    #     "config": "",
    #     "task_overrides": "",
    #     "overrides": "",
    # }
    # "example_name": {
    #     "description": "Example: describe your experiment for later.",
    #     "config": "Example: habitat_baselines/config/ant_v2/ppo_ant_v2_eval.yaml",
    #     "task_overrides": "Example: "task_overrides": " \"TASK.MEASUREMENTS [X_LOCATION]\"", #NOTE: you need the quotation marks here.
    #     "overrides": "Example: CHECKPOINT_FOLDER 'data/new_checkpoints_video_enabled_3'"
    # },
    "x_loc_ant": {
        "description": "Reward cumulative progress in X direction.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [X_LOCATION,VECTOR_ROOT_DELTA]\"",
        "overrides": " RL.REWARD_MEASURE X_LOCATION RL.SUCCESS_MEASURE X_LOCATION",
    },
    "delta_x_loc_ant": {
        "description": "Reward delta root progress in X direction.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [X_LOCATION,VECTOR_ROOT_DELTA]\"",
        "overrides": " RL.REWARD_MEASURE VECTOR_ROOT_DELTA RL.SUCCESS_MEASURE VECTOR_ROOT_DELTA",
    },
    "joint_error_ant":{
        "description": "Linear penalty for target joint angle error.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [JOINT_STATE_ERROR]\"",
        "overrides": " RL.REWARD_MEASURE JOINT_STATE_ERROR RL.SUCCESS_MEASURE JOINT_STATE_ERROR",
    },
    "joint_max_error_ant":{
        "description": "Linear penalty for target joint angle error.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [JOINT_STATE_MAX_ERROR]\"",
        "overrides": " RL.REWARD_MEASURE JOINT_STATE_MAX_ERROR RL.SUCCESS_MEASURE JOINT_STATE_MAX_ERROR",
    },
    "composite":{
        "description": "Composite reward term.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        #NOTE: overrides here will change based on contents of the composite reward
        "task_overrides": " \"TASK.MEASUREMENTS [JOINT_STATE_ERROR,VECTOR_ROOT_DELTA,COMPOSITE_ANT_REWARD,ACTION_COST]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD",
    },
    #NOTE: (best) joint error hyper-parameter options (02/11)
    "joint_error_ant_high_learning_rate":{
        "description": "Linear penalty for target joint angle error.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [JOINT_STATE_ERROR]\"",
        "overrides": " RL.SUCCESS_MEASURE JOINT_STATE_ERROR RL.REWARD_MEASURE JOINT_STATE_ERROR RL.PPO.lr 3e-4",
    },
    "joint_error_ant_low_clip":{
        "description": "Linear penalty for target joint angle error.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [JOINT_STATE_ERROR]\"",
        "overrides": " RL.SUCCESS_MEASURE JOINT_STATE_ERROR RL.REWARD_MEASURE JOINT_STATE_ERROR RL.PPO.clip_param 0.1",
    },
    #NOTE: joint error hyper-parameter options round 2 (02/14)
    "joint_error_ant_high_learning_rate_decay":{
        "description": "Linear penalty for target joint angle error.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [JOINT_STATE_ERROR]\"",
        "overrides": " RL.SUCCESS_MEASURE JOINT_STATE_ERROR RL.REWARD_MEASURE JOINT_STATE_ERROR RL.PPO.lr 3e-4 RL.PPO.use_linear_lr_decay True",
    },
    "joint_error_ant_high_learning_rate_low_clip":{
        "description": "Linear penalty for target joint angle error.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [JOINT_STATE_ERROR]\"",
        "overrides": " RL.SUCCESS_MEASURE JOINT_STATE_ERROR RL.REWARD_MEASURE JOINT_STATE_ERROR RL.PPO.lr 3e-4 RL.PPO.clip_param 0.1",
    },
    "joint_error_ant_clip_decay":{
        "description": "Linear penalty for target joint angle error.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [JOINT_STATE_ERROR]\"",
        "overrides": " RL.SUCCESS_MEASURE JOINT_STATE_ERROR RL.REWARD_MEASURE JOINT_STATE_ERROR RL.PPO.use_linear_clip_decay True",
    },
    "joint_error_ant_clip_and_high_lr_decay":{
        "description": "Linear penalty for target joint angle error.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [JOINT_STATE_ERROR]\"",
        "overrides": " RL.SUCCESS_MEASURE JOINT_STATE_ERROR RL.REWARD_MEASURE JOINT_STATE_ERROR RL.PPO.use_linear_clip_decay True RL.PPO.lr 3e-4 RL.PPO.use_linear_lr_decay True",
    },
    #NOTE: example of setting up a custom composite reward function.
    "custom_composite":{
        "description": "Customizing the composite reward term via config.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [ACTION_COST,JOINT_STATE_ERROR,X_LOCATION,COMPOSITE_ANT_REWARD] TASK.COMPOSITE_ANT_REWARD.COMPONENTS [ACTION_COST,JOINT_STATE_ERROR,X_LOCATION] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,1.0,10.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD",
    },
    #NOTE: joint error hyper-parameter options round 3 (02/14) - trying different step sizes and deltas
    "joint_error_ant_use_log_std":{
        "description": "Use the log std action distribution as suggested by Hab2 reach task",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR]\"",
        "overrides": " RL.SUCCESS_MEASURE JOINT_STATE_ERROR RL.REWARD_MEASURE JOINT_STATE_ERROR RL.POLICY.ACTION_DIST.use_log_std True",
    },
    "joint_error_ant_use_log_std_low_clip_high_lr":{
        "description": "Use the log std action distribution as suggested by Hab2 reach task",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR]\"",
        "overrides": " RL.SUCCESS_MEASURE JOINT_STATE_ERROR RL.REWARD_MEASURE JOINT_STATE_ERROR RL.POLICY.ACTION_DIST.use_log_std True RL.PPO.clip_param 0.1 RL.PPO.lr 3e-4",
    },
    "joint_error_ant_test_change_architecture":{
        "description": "Try LSTM to see if it works for PPO",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR]\"",
        "overrides": " RL.SUCCESS_MEASURE JOINT_STATE_ERROR RL.REWARD_MEASURE JOINT_STATE_ERROR RL.DDPPO.rnn_type LSTM RL.DDPPO.num_recurrent_layers 2 RL.DDPPO.distrib_backend NCCL",
    },
    "joint_error_ant_product_low_clip":{
        "description": "Try product of normalized error.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [JOINT_STATE_PRODUCT_ERROR,JOINT_STATE_ERROR]\"",
        "overrides": " RL.SUCCESS_MEASURE JOINT_STATE_PRODUCT_ERROR RL.REWARD_MEASURE JOINT_STATE_PRODUCT_ERROR RL.PPO.clip_param 0.1",
    },
    "joint_error_ant_normalized_low_clip":{
        "description": "Try normalized error.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [JOINT_STATE_PRODUCT_ERROR,JOINT_STATE_ERROR] TASK.JOINT_STATE_ERROR.NORMALIZED True\"",
        "overrides": " RL.SUCCESS_MEASURE JOINT_STATE_ERROR RL.REWARD_MEASURE JOINT_STATE_ERROR RL.PPO.clip_param 0.1",
    },
    #NOTE: New measures - Vector alignment (orientation), Velocity Target (movement)
    #NOTE: (02/16) Experiments for determining best control frequency
    "delta_x_low_clip_10hz":{
        "description": "Try 10hz action control.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [VECTOR_ROOT_DELTA,X_LOCATION,UPRIGHT_ORIENTATION_DEVIATION_DELTA,FORWARD_ORIENTATION_DEVIATION_DELTA] SIMULATOR.CTRL_FREQ 10 ENVIRONMENT.MAX_EPISODE_STEPS 50 TASK.ACTIONS.LEG_ACTION.DELTA_POS_LIMIT 0.3\"",
        "overrides": " RL.SUCCESS_MEASURE VECTOR_ROOT_DELTA RL.REWARD_MEASURE VECTOR_ROOT_DELTA RL.PPO.clip_param 0.1",
    },
    "delta_x_low_clip_15hz":{
        "description": "Try 15hz action control.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [VECTOR_ROOT_DELTA,X_LOCATION,UPRIGHT_ORIENTATION_DEVIATION_DELTA,FORWARD_ORIENTATION_DEVIATION_DELTA] SIMULATOR.CTRL_FREQ 15 ENVIRONMENT.MAX_EPISODE_STEPS 75 TASK.ACTIONS.LEG_ACTION.DELTA_POS_LIMIT 0.2\"",
        "overrides": " RL.SUCCESS_MEASURE VECTOR_ROOT_DELTA RL.REWARD_MEASURE VECTOR_ROOT_DELTA RL.PPO.clip_param 0.1",
    },
    "delta_x_low_clip_20hz":{
        "description": "Try 20hz action control.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [VECTOR_ROOT_DELTA,X_LOCATION,UPRIGHT_ORIENTATION_DEVIATION_DELTA,FORWARD_ORIENTATION_DEVIATION_DELTA] SIMULATOR.CTRL_FREQ 20 ENVIRONMENT.MAX_EPISODE_STEPS 100 TASK.ACTIONS.LEG_ACTION.DELTA_POS_LIMIT 0.15\"",
        "overrides": " RL.SUCCESS_MEASURE VECTOR_ROOT_DELTA RL.REWARD_MEASURE VECTOR_ROOT_DELTA RL.PPO.clip_param 0.1",
    },
    "delta_x_low_clip_30hz":{
        "description": "Try 30hz action control.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [VECTOR_ROOT_DELTA,X_LOCATION,UPRIGHT_ORIENTATION_DEVIATION_DELTA,FORWARD_ORIENTATION_DEVIATION_DELTA] SIMULATOR.CTRL_FREQ 30 ENVIRONMENT.MAX_EPISODE_STEPS 150 TASK.ACTIONS.LEG_ACTION.DELTA_POS_LIMIT 0.1\"",
        "overrides": " RL.SUCCESS_MEASURE VECTOR_ROOT_DELTA RL.REWARD_MEASURE VECTOR_ROOT_DELTA RL.PPO.clip_param 0.1",
    },
    #NOTE: Testing orientation alignment & velocity
    "ant_orientation_weights_v1":{
        "description": "Try teaching the ant to orient with a constant vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [ACTION_COST,UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,VELOCITY_ALIGNMENT,SPEED_TARGET,COMPOSITE_ANT_REWARD] TASK.JOINT_STATE_ERROR.NORMALIZED True SIMULATOR.TARGET_VECTOR [0.6,0.0,0.8] TASK.COMPOSITE_ANT_REWARD.COMPONENTS [ACTION_COST,UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,VELOCITY_ALIGNMENT,SPEED_TARGET] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,1.0,1.0,1.0,1.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_weights_v2":{
        "description": "Try teaching the ant to orient with a constant vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [ACTION_COST,UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,VELOCITY_ALIGNMENT,SPEED_TARGET,COMPOSITE_ANT_REWARD] TASK.JOINT_STATE_ERROR.NORMALIZED True SIMULATOR.TARGET_VECTOR [0.6,0.0,0.8] TASK.COMPOSITE_ANT_REWARD.COMPONENTS [ACTION_COST,UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,VELOCITY_ALIGNMENT,SPEED_TARGET] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,5.0,5.0,1.0,5.0,5.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_weights_v3":{
        "description": "Try teaching the ant to orient with a constant vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [ACTION_COST,UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,VELOCITY_ALIGNMENT,SPEED_TARGET,COMPOSITE_ANT_REWARD] TASK.JOINT_STATE_ERROR.NORMALIZED True SIMULATOR.TARGET_VECTOR [0.6,0.0,0.8] TASK.COMPOSITE_ANT_REWARD.COMPONENTS [ACTION_COST,UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,VELOCITY_ALIGNMENT,SPEED_TARGET] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [5.0,10.0,5.0,1.0,5.0,5.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_weights_v4":{
        "description": "Try teaching the ant to orient with a constant vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [ACTION_COST,UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,VELOCITY_ALIGNMENT,SPEED_TARGET,COMPOSITE_ANT_REWARD] TASK.JOINT_STATE_ERROR.NORMALIZED True SIMULATOR.TARGET_VECTOR [0.6,0.0,0.8] TASK.COMPOSITE_ANT_REWARD.COMPONENTS [ACTION_COST,UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,VELOCITY_ALIGNMENT,SPEED_TARGET] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [5.0,2.0,2.0,1.0,5.0,5.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    # JUST testing orientation alignment
    "ant_orientation_alignment":{
        "description": "Try teaching the ant to orient with a constant vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,COMPOSITE_ANT_REWARD] TASK.JOINT_STATE_ERROR.NORMALIZED True SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,4.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1 RL.PPO.lr 3e-4",
    },
    "ant_orientation_alignment_v2":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,COMPOSITE_ANT_REWARD] TASK.JOINT_STATE_ERROR.NORMALIZED True SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,4.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1 RL.PPO.lr 3e-4",
    },
    "ant_orientation_alignment_v5":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,COMPOSITE_ANT_REWARD] TASK.JOINT_STATE_ERROR.NORMALIZED True SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [2.0,10.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
     "ant_orientation_alignment_v4":{
        "description": "Try teaching the ant to orient with a constant vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,COMPOSITE_ANT_REWARD] TASK.JOINT_STATE_ERROR.NORMALIZED True SIMULATOR.TARGET_VECTOR [-1.0,0.0,0.0] TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [2.0,10.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    # ^ Not much luck
    # Trying random joint state regression tasks
    "joint_error_random_target_v1":{
        "description": "Try normalized error.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [JOINT_STATE_PRODUCT_ERROR,JOINT_STATE_ERROR] TASK.JOINT_STATE_ERROR.NORMALIZED True SIMULATOR.LEG_TARGET_STATE \"RANDOM\"\"",
        "overrides": " RL.SUCCESS_MEASURE JOINT_STATE_ERROR RL.REWARD_MEASURE JOINT_STATE_ERROR RL.PPO.clip_param 0.1",
    },
    
    # Implemented a baseline ant gait, action controller is deviation from this gait
    # Now we want to optimize this gait to maximize smoothness
    "ant_gait_optimization_v1":{
        "description": "Optimizing ant gait.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [VELOCITY_ALIGNMENT,ORTHOGONAL_VELOCITY,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,COMPOSITE_ANT_REWARD] TASK.COMPOSITE_ANT_REWARD.COMPONENTS [VELOCITY_ALIGNMENT,ORTHOGONAL_VELOCITY,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,1.0,1.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_gait_optimization_v2":{
        "description": "Optimizing ant gait.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [VELOCITY_ALIGNMENT,ORTHOGONAL_VELOCITY,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,COMPOSITE_ANT_REWARD] TASK.COMPOSITE_ANT_REWARD.COMPONENTS [ORTHOGONAL_VELOCITY,FORWARD_ORIENTATION_DEVIATION_VALUE] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    # Now adding periodic time as an observation
    "ant_gait_optimization_v3":{
        "description": "Optimizing ant gait.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [VELOCITY_ALIGNMENT,ORTHOGONAL_VELOCITY,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,COMPOSITE_ANT_REWARD] TASK.COMPOSITE_ANT_REWARD.COMPONENTS [VELOCITY_ALIGNMENT,ORTHOGONAL_VELOCITY,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [3.0,3.0,1.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    # Now aiming for a target orientation
    "ant_orientation_alignment__withgait_v1":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,COMPOSITE_ANT_REWARD] TASK.JOINT_STATE_ERROR.NORMALIZED True SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [2.0,2.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1 RL.PPO.lr 3e-4",
    },
    # Now adding action smoothness, fixed a few bugs, added action history as well as joint position history to observational space
    "ant_orientation_alignment__withgait_v2":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD] TASK.JOINT_STATE_ERROR.NORMALIZED True SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,ACTION_SMOOTHNESS] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,2.0,2.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    # Issues should be fixed, attempting several experiments
    "ant_orientation_alignment__withgait_v3":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD] SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,ACTION_SMOOTHNESS] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,2.0,2.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_alignment__withgait_v4":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD] SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,ACTION_SMOOTHNESS] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,5.0,2.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_alignment__withgait_v5":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD] SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,ACTION_SMOOTHNESS] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,2.0,2.0,4.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_alignment__withgait_v6":{
        "description": "Try teaching the ant to orient with a variable vector. Using squared forward alignment measure.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,ACTION_COST,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD] SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,ACTION_COST,ACTION_SMOOTHNESS] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,2.0,2.0,2.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_alignment__withgait_v7":{
        "description": "Try teaching the ant to orient with a variable vector. Using high LR",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,ACTION_COST,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD] SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,ACTION_COST,ACTION_SMOOTHNESS] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,2.0,2.0,2.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.lr 3e-4",
    },
    "ant_orientation_alignment__withgait_v8":{
        "description": "Try teaching the ant to orient with a variable vector. Using squared forward alignment measure. Lower action magnitude.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,ACTION_COST,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD] SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,ACTION_COST,ACTION_SMOOTHNESS] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,2.0,2.0,2.0,1.0] TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.2\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_alignment__withgait_v9":{
        "description": "Try teaching the ant to orient with a variable vector. Using squared forward alignment measure. Lower action magnitude.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,ACTION_COST,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD] SIMULATOR.TARGET_VECTOR [0.0,0.0,1.0] TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,ACTION_COST,ACTION_SMOOTHNESS] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,2.0,2.0,2.0,1.0] TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.2\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_gait_action_cost_reduction":{
        "description": "Try teaching the ant to orient with a variable vector. Using squared forward alignment measure. Lower action magnitude.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD] SIMULATOR.TARGET_VECTOR [1.0,0.0,0.0] TASK.COMPOSITE_ANT_REWARD.COMPONENTS [ACTION_COST] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0] TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 1.0\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    #from above, v4 was best. Others too much. Reducing delta from 0.5 and trying with greater action cost penalty
    #python run_exp.py --exp ant_orientation_alignment__withgait_d0.1_v4 --type train
    "ant_orientation_alignment__withgait_d0.1_v4":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.1 TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD] SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,ACTION_SMOOTHNESS] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,5.0,2.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_alignment__withgait_d0.1_v8":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.1 TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD] SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,ACTION_SMOOTHNESS] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,5.0,10.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_alignment__withgait_d0.1_v9":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.1 TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD] SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,ACTION_SMOOTHNESS] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,5.0,20.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_alignment__withgait_d0.1_v10":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.1 TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD] SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST,ACTION_SMOOTHNESS] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,5.0,5.0,5.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    #try larger penalties without the low delta?
    "ant_orientation_alignment__withgait_d0.3_v4":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.3 TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD] SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,5.0,2.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_alignment__withgait_d0.3_v8":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.3 TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD] SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,5.0,10.0,1.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_alignment__withgait_d0.3_v10":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.3 TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD] SIMULATOR.TARGET_VECTOR \"RANDOM\" TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,5.0,5.0,5.0]\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_alignment__withgait_d0.3_smooth10_v4":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \""+
            "TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.3" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS 10" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS 10" + 
            " TASK.ACTION_SMOOTHNESS.WINDOW 10" + 
            " TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD]" +
            " SIMULATOR.TARGET_VECTOR \"RANDOM\""+
            " TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS]"+
            " TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,5.0,2.0,1.0]"+
            "\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_alignment__withgait_d0.3_smooth10_v10":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \""+
            "TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.3" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS 10" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS 10" + 
            " TASK.ACTION_SMOOTHNESS.WINDOW 10" + 
            " TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD]" +
            " SIMULATOR.TARGET_VECTOR \"RANDOM\""+
            " TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS]"+
            " TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,5.0,5.0,5.0]"+
            "\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_gait_action_cost_reduction_with_sum":{
        "description": "Try teaching the ant to orient with a variable vector. Using squared forward alignment measure. Lower action magnitude.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \"TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_PRODUCT,ACTION_COST_SUM,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD] SIMULATOR.TARGET_VECTOR [1.0,0.0,0.0] TASK.COMPOSITE_ANT_REWARD.COMPONENTS [ACTION_COST_PRODUCT,ACTION_COST_SUM] TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,1.0] TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 1.0\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    # Low standard deviation; attempting to train the ant to orient itself with a constant target vector
    "ant_orientation_alignment_withgait_turn_right_v1":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \""+
            "TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.3" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS 10" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS 10" + 
            " TASK.ACTION_SMOOTHNESS.WINDOW 10" + 
            " TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD]" +
            " SIMULATOR.TARGET_VECTOR [0.0,0.0,1.0]"+
            " TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS]"+
            " TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,2.0,2.0,1.0]"+
            "\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_alignment_withgait_turn_right_v2":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \""+
            "TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.3" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS 10" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS 10" + 
            " TASK.ACTION_SMOOTHNESS.WINDOW 10" + 
            " TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD]" +
            " SIMULATOR.TARGET_VECTOR [0.0,0.0,1.0]"+
            " TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS]"+
            " TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,4.0,2.0,1.0]"+
            "\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_alignment_withgait_turn_backwards_v1":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \""+
            "TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.3" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS 10" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS 10" + 
            " TASK.ACTION_SMOOTHNESS.WINDOW 10" + 
            " TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD]" +
            " SIMULATOR.TARGET_VECTOR [-1.0,0.0,0.0]"+
            " TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS]"+
            " TASK.COMPOSITE_ANT_REWARD.WEIGHTS [2.0,2.0,2.0,1.0]"+
            "\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_alignment_withgait_turn_backwards_v2":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \""+
            "TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.3" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS 10" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS 10" + 
            " TASK.ACTION_SMOOTHNESS.WINDOW 10" + 
            " TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD]" +
            " SIMULATOR.TARGET_VECTOR [-1.0,0.0,0.0]"+
            " TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,ACTION_COST_SUM,ACTION_SMOOTHNESS]"+
            " TASK.COMPOSITE_ANT_REWARD.WEIGHTS [2.0,4.0,2.0,1.0]"+
            "\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    # Now experimenting with increasing Joint Delta Limit, Decreasing std more, and training the ant to learn the natural gait
    "ant_orientation_alignment_withgait_turn_right_v3":{
        "description": "Try teaching the ant to orient with a constant vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \""+
            "TASK.POSSIBLE_ACTIONS [LEG_ACTION_GAIT_DEVIATION]"
            " TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.5" + 
            #" RL.POLICY.ACTION_DIST.max_std = 0.075" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS 10" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS 10" + 
            " TASK.ACTION_SMOOTHNESS.WINDOW 10" + 
            " TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,ACTION_COST_SUM,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD]" +
            " SIMULATOR.TARGET_VECTOR [0.0,0.0,1.0]"+
            " TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,ACTION_COST_SUM,ACTION_SMOOTHNESS]"+
            " TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,2.0,2.0,2.0,1.0]"+
            "\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_alignment_withgait_turn_right_v4":{
        "description": "Try teaching the ant to orient with a constant vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \""+
            "TASK.POSSIBLE_ACTIONS [LEG_ACTION_GAIT_DEVIATION]"
            " TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.8" + 
            #" RL.POLICY.ACTION_DIST.max_std = 0.05" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS 10" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS 10" + 
            " TASK.ACTION_SMOOTHNESS.WINDOW 10" + 
            " TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,ACTION_COST_SUM,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD]" +
            " SIMULATOR.TARGET_VECTOR [0.0,0.0,1.0]"+
            " TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,ACTION_COST_SUM,ACTION_SMOOTHNESS]"+
            " TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,2.0,2.0,2.0,1.0]"+
            "\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_orientation_alignment_withgait_turn_right_v3":{
        "description": "Try teaching the ant to orient with a constant vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \""+
            "TASK.POSSIBLE_ACTIONS [LEG_ACTION_GAIT_DEVIATION]"
            " TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT 0.5" + 
            #" RL.POLICY.ACTION_DIST.max_std = 0.075" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS 10" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS 10" + 
            " TASK.ACTION_SMOOTHNESS.WINDOW 10" + 
            " TASK.MEASUREMENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,ACTION_COST_SUM,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD]" +
            " SIMULATOR.TARGET_VECTOR [0.0,0.0,1.0]"+
            " TASK.COMPOSITE_ANT_REWARD.COMPONENTS [UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,ACTION_COST_SUM,ACTION_SMOOTHNESS]"+
            " TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,2.0,4.0,2.0,1.0]"+
            "\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_train_gait_v1":{
        "description": "Try teaching the ant to walk with a natural gait.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \""+
            "TASK.POSSIBLE_ACTIONS [LEG_ACTION]"
            " TASK.ACTIONS.LEG_ACTION.DELTA_POS_LIMIT 0.3" + 
            #" RL.POLICY.ACTION_DIST.max_std = 0.1" + 
            " SIMULATOR.LEG_TARGET_STATE \"NATURAL_GAIT\""
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS 10" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS 10" + 
            " TASK.ACTION_SMOOTHNESS.WINDOW 10" + 
            " TASK.MEASUREMENTS [JOINT_STATE_ERROR,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD]" +
            " SIMULATOR.TARGET_VECTOR [0.0,0.0,1.0]"+
            " TASK.COMPOSITE_ANT_REWARD.COMPONENTS [JOINT_STATE_ERROR,ACTION_SMOOTHNESS]"+
            " TASK.COMPOSITE_ANT_REWARD.WEIGHTS [1.0,1.0]"+
            "\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
    "ant_train_gait_v2":{
        "description": "Try teaching the ant to orient with a variable vector.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": " \""+
            "TASK.POSSIBLE_ACTIONS [LEG_ACTION]"
            " TASK.ACTIONS.LEG_ACTION.DELTA_POS_LIMIT 0.3" + 
            #" RL.POLICY.ACTION_DIST.max_std = 0.1" + 
            " SIMULATOR.LEG_TARGET_STATE \"NATURAL_GAIT\""
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS 10" + 
            " TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS 10" + 
            " TASK.ACTION_SMOOTHNESS.WINDOW 10" + 
            " TASK.MEASUREMENTS [JOINT_STATE_ERROR,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD]" +
            " SIMULATOR.TARGET_VECTOR [0.0,0.0,1.0]"+
            " TASK.COMPOSITE_ANT_REWARD.COMPONENTS [JOINT_STATE_ERROR,ACTION_SMOOTHNESS]"+
            " TASK.COMPOSITE_ANT_REWARD.WEIGHTS [2.0,1.0]"+
            "\"",
        "overrides": " RL.SUCCESS_MEASURE COMPOSITE_ANT_REWARD RL.REWARD_MEASURE COMPOSITE_ANT_REWARD RL.PPO.clip_param 0.1",
    },
}


run_types = ["eval", "train"]
run_base = "python -u habitat_baselines/run.py"


def run(experiment=None, run_type="train", testing=False):
    assert experiment in experiments
    assert run_type in run_types

    exp_info = experiments[experiment]

    full_command = run_base
    
    #add the config
    full_command += " --exp-config " + exp_info["config"]
    #add the type
    full_command += " --run-type " + run_type
    
    #add task overrides
    if "task_overrides" in exp_info:
        task_overrides = exp_info["task_overrides"]
        full_command += " --task-overrides" + task_overrides

    overrides = ""

    #add overrides for exp name
    #NOTE: storing results in data/checkpoints/<expriment_name>/ and data/tb/<expriment_name>/
    overrides += " TENSORBOARD_DIR data/tb/" + experiment + "/"
    overrides += " CHECKPOINT_FOLDER data/checkpoints/" + experiment + "/"
    overrides += exp_info["overrides"]

    #add known type overrides for eval
    if run_type == "eval":
        #overrides += " VIDEO_OPTION "
        overrides += " EVAL_CKPT_PATH_DIR data/checkpoints/" + experiment + "/"
        overrides += " VIDEO_DIR data/videos/" + experiment + "/"
        #NOTE: this adds the extra sensor for visualization
        overrides += " SENSORS ['THIRD_SENSOR']"
        if not testing:
            #NOTE: number of videos/data_points you want
            overrides += " NUM_ENVIRONMENTS 3"

    #settings for easier debugging
    if testing:
        overrides += " NUM_ENVIRONMENTS 1"
        overrides += " RL.PPO.num_mini_batch 1"

    #add the overrides
    full_command += overrides

    #NOTE: comment out the final line to check before running if in doubt
    print(full_command)
    os.system(full_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="name of the experiment as defined in the experiments dict",
    )
    parser.add_argument("--test", action="store_true", help="If provided, sets mini-batches to 1 and environments to 1 for easy debugging.")

    args = parser.parse_args()
    run(experiment=args.exp, run_type=args.type, testing=args.test)