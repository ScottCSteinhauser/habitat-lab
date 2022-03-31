import os
import argparse
import json
import copy
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
    #     "config": "", #optional
    #     "task_overrides": {}, #overrides to the task config
    #     "overrides": {}, #overrides to the learning config
    # }
    "ant_train_gait_orientation_template":{
        "description": "Try teaching the ant to walk with a natural gait and turn to the right.",
        "task_overrides": {
            "TASK.POSSIBLE_ACTIONS": "[LEG_ACTION_ABS]",
            "SIMULATOR.LEG_TARGET_STATE": "\"NATURAL_GAIT\"",
            "TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS": "10",
            "TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS": "10",
            "TASK.ACTION_SMOOTHNESS.WINDOW": "10",
            "TASK.MEASUREMENTS": "[UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS,ORIENTATION_TERMINATE,COMPOSITE_ANT_REWARD]",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0,1.0,1.0,1.0]",
            "TASK.COMPOSITE_ANT_REWARD.ADDITIONAL_REQUIREMENTS": "[ORIENTATION_TERMINATE]"
        },
        "overrides": {
            "RL.SUCCESS_MEASURE": "COMPOSITE_ANT_REWARD",
            "RL.REWARD_MEASURE": "COMPOSITE_ANT_REWARD",
            "RL.PPO.clip_param": "0.1",
            "RL.POLICY.ACTION_DIST.max_std": "0.1"
        }
    },
    "ant_train_gait_abscontroller_base":{
        "description": "Try teaching the ant to walk with a natural gait and absolute position actions.",
        "task_overrides": {
            "TASK.POSSIBLE_ACTIONS": "[LEG_ACTION_ABS]",
            "SIMULATOR.LEG_TARGET_STATE": "\"NATURAL_GAIT\"",
            "TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS": "10",
            "TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS": "10", 
            "TASK.ACTION_SMOOTHNESS.WINDOW": "10", 
            "TASK.MEASUREMENTS": "[DEEP_MIMIC_POSE_REWARD,DEEP_MIMIC_JOINT_VELOCITY_REWARD,DEEP_MIMIC_END_EFFECTOR_POSITION_REWARD,DEEP_MIMIC_TARGET_HEADING,X_LOCATION,UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS,ORIENTATION_TERMINATE,DEEP_MIMIC_POSE_COMPOSITE_ANT_REWARD,COMPOSITE_ANT_REWARD]",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0]",
            "TASK.COMPOSITE_ANT_REWARD.ADDITIONAL_REQUIREMENTS": "[ORIENTATION_TERMINATE]",
        },
        "overrides": {
            "RL.SUCCESS_MEASURE": "COMPOSITE_ANT_REWARD",
            "RL.REWARD_MEASURE": "COMPOSITE_ANT_REWARD",
            "RL.PPO.clip_param": "0.1",
            "RL.POLICY.ACTION_DIST.max_std": "0.1",
        }
    },
    "ant_flat_position_base":{
        "description": "Try teaching the ant to flatten itself.",
        "task_overrides": {
            "TASK.POSSIBLE_ACTIONS": "[LEG_ACTION_ABS]",
            "SIMULATOR.LEG_TARGET_STATE": "[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]",
            "TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS": "10",
            "TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS": "10", 
            "TASK.ACTION_SMOOTHNESS.WINDOW": "10", 
            "TASK.MEASUREMENTS": "[UPRIGHT_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS,ORIENTATION_TERMINATE,COMPOSITE_ANT_REWARD]",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0]",
            "TASK.COMPOSITE_ANT_REWARD.ADDITIONAL_REQUIREMENTS": "[ORIENTATION_TERMINATE]",
        },
        "overrides": {
            "RL.SUCCESS_MEASURE": "COMPOSITE_ANT_REWARD",
            "RL.REWARD_MEASURE": "COMPOSITE_ANT_REWARD",
            "RL.PPO.clip_param": "0.1",
            "RL.POLICY.ACTION_DIST.max_std": "0.1",
        }
    },
    "ant_delta_x_base":{
        "description": "Try teaching the ant to walk forward with no guidance.",
        "task_overrides": {
            "TASK.POSSIBLE_ACTIONS": "[LEG_ACTION]",
            "TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS": "10",
            "TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS": "10", 
            "TASK.ACTION_SMOOTHNESS.WINDOW": "10", 
            "SIMULATOR.TARGET_VECTOR": "[1.0,0.0,0.0]",
            "TASK.MEASUREMENTS": "[UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,VECTOR_ROOT_DELTA,ACTION_SMOOTHNESS,ORIENTATION_TERMINATE,COMPOSITE_ANT_REWARD]",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,VECTOR_ROOT_DELTA,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,5.0,1.0]",
            "TASK.COMPOSITE_ANT_REWARD.ADDITIONAL_REQUIREMENTS": "[ORIENTATION_TERMINATE]",        
            },
        "overrides": {
            "RL.SUCCESS_MEASURE": "COMPOSITE_ANT_REWARD",
            "RL.REWARD_MEASURE": "COMPOSITE_ANT_REWARD",
            "RL.PPO.clip_param": "0.1",
            "RL.POLICY.ACTION_DIST.max_std": "0.1",
        }
    },
    "ant_delta_x_abs_base":{
        "description": "Try teaching the ant to walk forward with no guidance.",
        "task_overrides": {
            "TASK.POSSIBLE_ACTIONS": "[LEG_ACTION_ABS]",
            "TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS": "10",
            "TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS": "10", 
            "TASK.ACTION_SMOOTHNESS.WINDOW": "10", 
            "SIMULATOR.TARGET_VECTOR": "[1.0,0.0,0.0]",
            "TASK.MEASUREMENTS": "[UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,VECTOR_ROOT_DELTA,X_LOCATION,ACTION_SMOOTHNESS,ORIENTATION_TERMINATE,COMPOSITE_ANT_REWARD]",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[VECTOR_ROOT_DELTA,X_LOCATION,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[100.0,1.0,1.0]",
            "TASK.COMPOSITE_ANT_REWARD.ADDITIONAL_REQUIREMENTS": "[ORIENTATION_TERMINATE]",        
            },
        "overrides": {
            "RL.SUCCESS_MEASURE": "COMPOSITE_ANT_REWARD",
            "RL.REWARD_MEASURE": "COMPOSITE_ANT_REWARD",
            "RL.PPO.clip_param": "0.1",
            "RL.POLICY.ACTION_DIST.max_std": "0.1",
        }
    },
    
    "ant_delta_x_rel_base":{
        "description": "Try teaching the ant to walk forward with no guidance, using relative position controller.",
        "task_overrides": {
            "TASK.POSSIBLE_ACTIONS": "[LEG_ACTION]",
            "TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS": "10",
            "TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS": "10", 
            "TASK.ACTION_SMOOTHNESS.WINDOW": "10", 
            "SIMULATOR.TARGET_VECTOR": "[1.0,0.0,0.0]",
            "TASK.MEASUREMENTS": "[UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,VECTOR_ROOT_DELTA,X_LOCATION,ACTION_SMOOTHNESS,ORIENTATION_TERMINATE,COMPOSITE_ANT_REWARD]",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[X_LOCATION,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[2.0,1.0]",
            "TASK.COMPOSITE_ANT_REWARD.ADDITIONAL_REQUIREMENTS": "[ORIENTATION_TERMINATE]",        
            },
        "overrides": {
            "RL.SUCCESS_MEASURE": "COMPOSITE_ANT_REWARD",
            "RL.REWARD_MEASURE": "COMPOSITE_ANT_REWARD",
            "RL.PPO.clip_param": "0.1",
            "RL.POLICY.ACTION_DIST.max_std": "0.04",
        }
    },

}

#variations of base experiments:
experiment_variations: Dict[str, Dict[str,str]] = {
    #abs gait imitation max_std variations
    "ant_train_gait_abscontroller_var_v1":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{},
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.01"},
    },
    "ant_train_gait_abscontroller_var_v2":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{},
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.025"},
    },
    "ant_train_gait_abscontroller_var_v3":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{},
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.04"},
    },
    "ant_train_gait_abscontroller_var_v4":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{},
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.06"},
    },
    "ant_train_gait_abscontroller_var_v5":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{},
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.075"},
    },
    #rel gait imitation + turning variations
    "ant_train_gait_orientation_rel_v1":{
        "base_experiment": "ant_train_gait_orientation_template",
        "task_overrides":{"TASK.POSSIBLE_ACTIONS": "[LEG_ACTION]"},
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.1"},
    },
    "ant_train_gait_orientation_rel_v2":{
        "base_experiment": "ant_train_gait_orientation_template",
        "task_overrides":{
            "TASK.POSSIBLE_ACTIONS": "[LEG_ACTION]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,2.0,2.0,1.0,1.0,1.0]",
            },
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.1"},
    },
    #rel gait imitation + turning variations (hypothesis that 0.02 is good max_std)
    "ant_train_gait_orientation_abs_v1":{
        "base_experiment": "ant_train_gait_orientation_template",
        "task_overrides":{},
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.02"},
    },
    "ant_train_gait_orientation_abs_v2":{
        "base_experiment": "ant_train_gait_orientation_template",
        "task_overrides":{
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,2.0,2.0,1.0,1.0,1.0]",
        },
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.02"},
    },
    
    # -- experiments for undergraduate thesis --
    
    # Joint Regression to flat state
    # Try using Rel position controller
    "ant_flat_position_base_v1":{
        "base_experiment": "ant_flat_position_base",
        "task_overrides":{
            "TASK.POSSIBLE_ACTIONS": "[LEG_ACTION]",
        },
        "overrides": {},
    },
    
    # X Delta
    "ant_delta_x_rel_v1":{
        "base_experiment": "ant_delta_x_base",
        "task_overrides":{},
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.07"},
    },
    "ant_delta_x_rel_v2":{
        "base_experiment": "ant_delta_x_base",
        "task_overrides":{},
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.04"},
    },
    
    # use abs controller
    "ant_delta_x_abs_v1":{
        "base_experiment": "ant_delta_x_base",
        "task_overrides":{
            "TASK.POSSIBLE_ACTIONS": "[LEG_ACTION_ABS]",
        },
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.07"},
    },
    "ant_delta_x_abs_v2":{
        "base_experiment": "ant_delta_x_base",
        "task_overrides":{
            "TASK.POSSIBLE_ACTIONS": "[LEG_ACTION_ABS]",
        },
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.04"},
    },
    
    # Need to change vector_root_delta to x_delta
    "ant_flat_position_base_v2":{
        "base_experiment": "ant_flat_position_base",
        "task_overrides":{
            "TASK.JOINT_STATE_ERROR.NORMALIZED": "True"
        },
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.04"},
    },
    
    "ant_flat_position_base_v3":{
        "base_experiment": "ant_flat_position_base",
        "task_overrides":{
            "TASK.JOINT_STATE_ERROR.NORMALIZED": "True"
        },
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.01"},
    },
    
    # use abs controller
    "ant_delta_x_abs_v3":{
        "base_experiment": "ant_delta_x_abs_base",
        "task_overrides":{},
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.03"},
    },
    "ant_delta_x_abs_v4":{
        "base_experiment": "ant_delta_x_abs_base",
        "task_overrides":{},
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.04"},
    },
    "ant_delta_x_abs_v5":{
        "base_experiment": "ant_delta_x_abs_base",
        "task_overrides":{},
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.05"},
    },
    # Try even smoother actions
    "ant_delta_x_abs_v6":{
        "base_experiment": "ant_delta_x_abs_base",
        "task_overrides":{
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[100.0,1.0,5.0]",    
        },
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.04"},
    },
    
    # random joint position regression
    "ant_random_position_v1":{
        "base_experiment": "ant_flat_position_base",
        "task_overrides":{
            "SIMULATOR.LEG_TARGET_STATE": "\"RANDOM\"",
            "TASK.JOINT_STATE_ERROR.NORMALIZED": "True",
            "TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTIVE_TERMS": "[\"JOINT_POS\",\"JOINT_MOTOR_POS\",\"JOINT_VEL\",\"JOINT_TARGET\",\"NEXT_JOINT_TARGET\",\"JOINT_POSITION_HISTORY\",\"ACTION_HISTORY\"]",
        },
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.04"},
    },
    
    # Ant walking with no prior using the relative controller
    "ant_train_gait_abscontroller_corridor_v1":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "False",
            },
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.04"},
    },
    
    "ant_train_gait_abscontroller_corridor_v2":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "True",
            },
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.04"},
    },
    
    # With Forward X
    "ant_train_gait_abscontroller_corridor_v3":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "False",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[X_LOCATION,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0,1.0]",
            },
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.04"},
    },
    
    "ant_train_gait_abscontroller_corridor_v4":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "True",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[X_LOCATION,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0,1.0]",
            },
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.04"},
    },
    
    # With Forward Alignment
    "ant_train_gait_abscontroller_corridor_v5":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "False",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0,1.0]",
            },
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.04"},
    },
    
    "ant_train_gait_abscontroller_corridor_v6":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "True",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[X_LOCATION,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0,1.0,1.0]",
            },
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.04"},
    },
    
    # With Forward X and Forward Alignment
    "ant_train_gait_abscontroller_corridor_v7":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "False",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[X_LOCATION,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0,1.0,1.0]",
            },
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.04"},
    },
    
    "ant_train_gait_abscontroller_corridor_v8":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "True",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0,1.0]",
            },
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.04"},
    },
    
    # TESTING NEW SENSOR
    "ant_train_gait_abscontroller_corridor_v9_TEST":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "TASK.JOINT_STATE_ERROR.NORMALIZED": "True",
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "True",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0,1.0]",
            },
        "overrides": {"RL.POLICY.ACTION_DIST.max_std": "0.04"},
    },
    
    # Experiments with vision introduced
    "ant_train_gait_abscontroller_corridor_vision_v1":{ 
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "ENVIRONMENT.MAX_EPISODE_STEPS": "300",
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "False",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0]",
            },
        "overrides": {
            "RL.POLICY.ACTION_DIST.max_std": "0.04",
            "SENSORS": "['HEAD_RGB_SENSOR']",
            "RL.PPO.num_steps": "600" 
            },
    },
    "ant_train_gait_abscontroller_corridor_vision_v2":{ 
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "ENVIRONMENT.MAX_EPISODE_STEPS": "300",
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "True",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0]",
            },
        "overrides": {
            "RL.POLICY.ACTION_DIST.max_std": "0.04",
            "SENSORS": "['HEAD_RGB_SENSOR']",
            "RL.PPO.num_steps": "600" 
            },
    },
    
    # with forward X
    "ant_train_gait_abscontroller_corridor_vision_v3":{ 
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "ENVIRONMENT.MAX_EPISODE_STEPS": "300",
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "False",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[X_LOCATION,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0,1.0]",
            },
        "overrides": {
            "RL.POLICY.ACTION_DIST.max_std": "0.04",
            "SENSORS": "['HEAD_RGB_SENSOR']",
            "RL.PPO.num_steps": "600" 
            },
    },
    "ant_train_gait_abscontroller_corridor_vision_v4":{ 
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "ENVIRONMENT.MAX_EPISODE_STEPS": "300",
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "True",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[X_LOCATION,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0,1.0]",
            },
        "overrides": {
            "RL.POLICY.ACTION_DIST.max_std": "0.04",
            "SENSORS": "['HEAD_RGB_SENSOR']",
            "RL.PPO.num_steps": "600" 
            },
    },
    
    # with forward alignment
    "ant_train_gait_abscontroller_corridor_vision_v5":{ 
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "ENVIRONMENT.MAX_EPISODE_STEPS": "300",
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "False",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0,1.0]",
            },
        "overrides": {
            "RL.POLICY.ACTION_DIST.max_std": "0.04",
            "SENSORS": "['HEAD_RGB_SENSOR']",
            "RL.PPO.num_steps": "600" 
            },
    },
    "ant_train_gait_abscontroller_corridor_vision_v6":{ 
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "ENVIRONMENT.MAX_EPISODE_STEPS": "300",
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "True",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0,1.0]",
            },
        "overrides": {
            "RL.POLICY.ACTION_DIST.max_std": "0.04",
            "SENSORS": "['HEAD_RGB_SENSOR']",
            "RL.PPO.num_steps": "600" 
            },
    },
    
    # with x location and forward alignment
    "ant_train_gait_abscontroller_corridor_vision_v7":{ 
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "ENVIRONMENT.MAX_EPISODE_STEPS": "300",
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "False",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[X_LOCATION,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0,1.0,1.0]",
            },
        "overrides": {
            "RL.POLICY.ACTION_DIST.max_std": "0.04",
            "SENSORS": "['HEAD_RGB_SENSOR']",
            "RL.PPO.num_steps": "600" 
            },
    },
    "ant_train_gait_abscontroller_corridor_vision_v8":{ 
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "ENVIRONMENT.MAX_EPISODE_STEPS": "300",
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "True",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[X_LOCATION,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0,1.0,1.0]",
            },
        "overrides": {
            "RL.POLICY.ACTION_DIST.max_std": "0.04",
            "SENSORS": "['HEAD_RGB_SENSOR']",
            "RL.PPO.num_steps": "600"
            },
    },
    
    # Testing Deep Mimic reward formulation
    # Just pose mimic
    "ant_train_gait_abscontroller_corridor_deep_mimic_v1":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "ENVIRONMENT.MAX_EPISODE_STEPS": "300",
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "False",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[DEEP_MIMIC_POSE_COMPOSITE_ANT_REWARD]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0]",
            },
        "overrides": {
                "RL.POLICY.ACTION_DIST.max_std": "0.04",
                "RL.PPO.num_steps": "600", 
            },
    },
    
    "ant_train_gait_abscontroller_corridor_deep_mimic_v2":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "ENVIRONMENT.MAX_EPISODE_STEPS": "300",
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "True",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[DEEP_MIMIC_POSE_COMPOSITE_ANT_REWARD]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0]",
            },
        "overrides": {
                "RL.POLICY.ACTION_DIST.max_std": "0.04",
                "RL.PPO.num_steps": "600", 
            },
    },
    # Include target direction
    "ant_train_gait_abscontroller_corridor_deep_mimic_v3":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "ENVIRONMENT.MAX_EPISODE_STEPS": "300",
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "False",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[DEEP_MIMIC_POSE_COMPOSITE_ANT_REWARD,DEEP_MIMIC_TARGET_HEADING]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[0.7,0.3]",
            },
        "overrides": {
                "RL.POLICY.ACTION_DIST.max_std": "0.04",
                "RL.PPO.num_steps": "600", 
            },
    },
    
    "ant_train_gait_abscontroller_corridor_deep_mimic_v4":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "ENVIRONMENT.MAX_EPISODE_STEPS": "300",
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "True",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[DEEP_MIMIC_POSE_COMPOSITE_ANT_REWARD,DEEP_MIMIC_TARGET_HEADING]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[0.7,0.3]",
            },
        "overrides": {
                "RL.POLICY.ACTION_DIST.max_std": "0.04",
                "RL.PPO.num_steps": "600", 
            },
    },
    # With Vision
    "ant_train_gait_abscontroller_corridor_deep_mimic_v5":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "ENVIRONMENT.MAX_EPISODE_STEPS": "300",
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "False",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[DEEP_MIMIC_POSE_COMPOSITE_ANT_REWARD,DEEP_MIMIC_TARGET_HEADING]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[0.7,0.3]",
            },
        "overrides": {
                "RL.POLICY.ACTION_DIST.max_std": "0.04",
                "SENSORS": "['HEAD_RGB_SENSOR']",
                "RL.PPO.num_steps": "600", 
            },
    },
    
    "ant_train_gait_abscontroller_corridor_deep_mimic_v6":{
        "base_experiment": "ant_train_gait_abscontroller_base",
        "task_overrides":{
            "ENVIRONMENT.MAX_EPISODE_STEPS": "300",
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "True",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[DEEP_MIMIC_POSE_COMPOSITE_ANT_REWARD,DEEP_MIMIC_TARGET_HEADING]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[0.7,0.3]",
            },
        "overrides": {
                "RL.POLICY.ACTION_DIST.max_std": "0.04",
                "SENSORS": "['HEAD_RGB_SENSOR']",
                "RL.PPO.num_steps": "600", 
            },
    },
    
}

#merge variations into experiments
for var,var_info in experiment_variations.items():
    experiments[var] = copy.deepcopy(experiments[var_info["base_experiment"]])
    for override_type in ("task_overrides", "overrides"):
        for key,val in var_info[override_type].items():
            experiments[var][override_type][key] = val

#NOTE: use this to check the final dict results:
#print(json.dumps(experiments, indent=4))
#exit()

run_types = ["eval", "train"]
run_base = "python -u habitat_baselines/run.py"


def run(experiment=None, run_type="train", testing=False, quick_eval=False):
    assert experiment in experiments
    assert run_type in run_types

    exp_info = experiments[experiment]

    full_command = run_base
    
    #add the config
    if "config" in exp_info:
        full_command += " --exp-config " + exp_info["config"]
    else:
        #default config
        full_command += " --exp-config habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml"
    #add the type
    full_command += " --run-type " + run_type
    
    #add task overrides
    task_overrides = exp_info["task_overrides"]
    full_command += " --task-overrides \""
    for key, val in task_overrides.items():
        full_command += " " + key + " " + val
    full_command += "\""

    overrides = ""
    #add overrides for exp name
    #NOTE: storing results in data/checkpoints/<expriment_name>/ and data/tb/<expriment_name>/
    overrides += " TENSORBOARD_DIR data/tb/" + experiment + "/"
    overrides += " CHECKPOINT_FOLDER data/checkpoints/" + experiment + "/"
    
    exp_overrides = exp_info["overrides"]
    for key, val in exp_overrides.items():
        overrides += " " + key + " " + val

    #add known type overrides for eval
    if run_type == "eval":
        #overrides += " VIDEO_OPTION "
        overrides += " EVAL_CKPT_PATH_DIR data/checkpoints/" + experiment + "/"
        overrides += " VIDEO_DIR data/videos/" + experiment + "/"
        #NOTE: this adds the extra sensor for visualization
        if 'HEAD_RGB_SENSOR' in overrides:
            # string replacement
            sides = overrides.split("SENSORS ['HEAD_RGB_SENSOR']")
            overrides = sides[0] + "SENSORS ['THIRD_SENSOR','HEAD_RGB_SENSOR']" + sides[1]
        else:
            overrides += " SENSORS ['THIRD_SENSOR']"
            
        
        if not testing:
            #NOTE: number of videos/data_points you want
            overrides += " NUM_ENVIRONMENTS 3"
        if quick_eval:
            overrides += " EVAL.CKPT_INC 10"

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
        "--get-cmds", 
        action="store_true",
        help="Instead of running, export a list of run commands for easy copy-paste."
    )
    parser.add_argument(
        "--type",
        choices=["train", "eval", "quick-eval"],
        required=True,
        help="run type of the experiment (train, eval, or quick-eval (increments of 10))",
    )
    parser.add_argument(
        "--exp",
        type=str,
        help="name of the experiment as defined in the experiments dict",
    )
    parser.add_argument("--test", action="store_true", help="If provided, sets mini-batches to 1 and environments to 1 for easy debugging.")

    args = parser.parse_args()

    if args.get_cmds:
        #print copy template run commands
        print("============================================================")
        print(" Run Command Templates: ")
        print("------------------------------------------------------------")
        run_cmd_prefix = "python run_exp.py --exp "
        run_cmd_postfix = " --type " + args.type 
        for exp_name in experiments.keys():
            print(run_cmd_prefix + exp_name + run_cmd_postfix)
        print("============================================================")
    else:
        run_type = "eval" if args.type in ["eval", "quick-eval"] else "train"
        run(experiment=args.exp, run_type=run_type, testing=args.test, quick_eval=(args.type == "quick-eval"))