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
    "ant_joint_state_regression_base":{
        "description": "Try teaching the ant to achieve a constant joint configuration.",
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
        },
    },
        
    "ant_move_forward_rel_pos_base":{
        "description": "Reward cumulative progress in X direction.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": {
            "TASK.POSSIBLE_ACTIONS": "[LEG_ACTION]",
            "SIMULATOR.LEG_TARGET_STATE": "[1.0,0.0,-1.0,0.0,1.0,0.0,-1.0,0.0]",
            "TASK.MEASUREMENTS": "[X_LOCATION,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,COMPOSITE_ANT_REWARD]",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[X_LOCATION,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0]",
        },
        "overrides": {
            "RL.SUCCESS_MEASURE": "COMPOSITE_ANT_REWARD",
            "RL.REWARD_MEASURE": "COMPOSITE_ANT_REWARD",
            "RL.PPO.clip_param": "0.1",
            "RL.PPO.lr": "3e-4",
            "RL.POLICY.ACTION_DIST.max_std": "0.1", #in case default changes in future
        },
    },

    "ant_train_gait_abscontroller_base":{
        "description": "Try teaching the ant to walk with a natural gait and absolute position actions.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": {
            "TASK.POSSIBLE_ACTIONS": "[LEG_ACTION_ABS]",
            "SIMULATOR.LEG_TARGET_STATE_MODE": "\"NATURAL_GAIT\"",
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
            "RL.POLICY.ACTION_DIST.max_std": "0.04",
        },
    },
    
    "ant_train_orientation_with_gait_deviation_base":{
        "description": "Try teaching the ant to orient with a constant vector using the gait deviation action controller.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": {
            "TASK.POSSIBLE_ACTIONS": "[LEG_ACTION_GAIT_DEVIATION]",
            "TASK.ACTIONS.LEG_ACTION_GAIT_DEVIATION.DELTA_POS_LIMIT": "0.5", 
            "TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS": "10", 
            "TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS": "10",
            "TASK.ACTION_SMOOTHNESS.WINDOW": "10",
            "TASK.MEASUREMENTS": "[UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,ACTION_COST_SUM,ACTION_SMOOTHNESS,COMPOSITE_ANT_REWARD]",
            "SIMULATOR.TARGET_VECTOR": "[0.0,0.0,1.0]",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE_SQUARED,ACTION_COST_SUM,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,2.0,4.0,2.0,1.0]",
        },   
        "overrides": {
            "RL.SUCCESS_MEASURE": "COMPOSITE_ANT_REWARD",
            "RL.REWARD_MEASURE": "COMPOSITE_ANT_REWARD",
            "RL.PPO.clip_param": "0.1",
        },
    },
    
    "ant_train_gait_abscontroller_corridor_vision_base":{ 
        "description": "Try teaching the ant to walk with a natural gait and absolute position actions, maintaining a forward orientation.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "task_overrides": {
            "TASK.POSSIBLE_ACTIONS": "[LEG_ACTION_ABS]",
            "SIMULATOR.LEG_TARGET_STATE_MODE": "\"NATURAL_GAIT\"",
            "ENVIRONMENT.MAX_EPISODE_STEPS": "300",
            "SIMULATOR.LOAD_CORRIDOR": "True",
            "SIMULATOR.LOAD_OBSTACLES": "False",
            "TASK.ANT_OBSERVATION_SPACE_SENSOR.ACTION_HISTORY.NUM_STEPS": "10",
            "TASK.ANT_OBSERVATION_SPACE_SENSOR.JOINT_POSITION_HISTORY.NUM_STEPS": "10", 
            "TASK.ACTION_SMOOTHNESS.WINDOW": "10", 
            "TASK.MEASUREMENTS": "[DEEP_MIMIC_POSE_REWARD,DEEP_MIMIC_JOINT_VELOCITY_REWARD,DEEP_MIMIC_END_EFFECTOR_POSITION_REWARD,DEEP_MIMIC_TARGET_HEADING,X_LOCATION,UPRIGHT_ORIENTATION_DEVIATION_VALUE,FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS,ORIENTATION_TERMINATE,DEEP_MIMIC_POSE_COMPOSITE_ANT_REWARD,COMPOSITE_ANT_REWARD]",
            "TASK.COMPOSITE_ANT_REWARD.COMPONENTS": "[FORWARD_ORIENTATION_DEVIATION_VALUE,JOINT_STATE_ERROR,JOINT_STATE_PRODUCT_ERROR,ACTION_SMOOTHNESS]",
            "TASK.COMPOSITE_ANT_REWARD.WEIGHTS": "[1.0,1.0,1.0,1.0]",
            "TASK.COMPOSITE_ANT_REWARD.ADDITIONAL_REQUIREMENTS": "[ORIENTATION_TERMINATE]",
        },
        "overrides": {
            "RL.SUCCESS_MEASURE": "COMPOSITE_ANT_REWARD",
            "RL.REWARD_MEASURE": "COMPOSITE_ANT_REWARD",
            "SENSORS": "['HEAD_RGB_SENSOR']",
            "RL.PPO.clip_param": "0.1",
            "RL.POLICY.ACTION_DIST.max_std": "0.04",
            "RL.PPO.num_steps": "600",
        },
    },
    
}

#variations of base experiments:
#NOTE: See the examples/template below for formatting
experiment_variations: Dict[str, Dict[str,str]] = {
    # "template_exp_key":{
    #     "base_experiment": "base_experiment_key",
    #     "task_overrides": {}, #overrides to the task config (duplicate entries override base["task_overrides"]["key"])
    #     "overrides": {}, #overrides to the learning config (duplicate entries override base["overrides"]["key"])
    # }
    "ant_joint_state_regression_new_target":{
        "base_experiment":"ant_joint_state_regression_base",
        "task_overrides": {"SIMULATOR.LEG_TARGET_STATE": "[0.0,-1.0,0.0,-1.0,0.0,1.0,0.0,1.0]"}, #rest standing pose
         "overrides":{}
    },
    "ant_move_forward_rel_pos_simple":{
        #reduce to X_LOCATION only
        "base_experiment":"ant_move_forward_rel_pos_base",
        "task_overrides": {
            "TASK.MEASUREMENTS": "[X_LOCATION]",
        },
        "overrides":{
            "RL.SUCCESS_MEASURE": "X_LOCATION",
            "RL.REWARD_MEASURE": "X_LOCATION",
        }
    },
    "ant_move_forward_rel_pos_simple_low_std":{
        #reduce to X_LOCATION only
        "base_experiment":"ant_move_forward_rel_pos_base",
        "task_overrides": {
            "TASK.MEASUREMENTS": "[X_LOCATION]",
        },
        "overrides":{
            "RL.SUCCESS_MEASURE": "X_LOCATION",
            "RL.REWARD_MEASURE": "X_LOCATION",
            "RL.POLICY.ACTION_DIST.max_std": "0.04",
        }
    },
    "ant_move_forward_abs_pos_simple":{
        #reduce to X_LOCATION only
        "base_experiment":"ant_move_forward_rel_pos_base",
        "task_overrides": {
            #abs position control
            "TASK.POSSIBLE_ACTIONS": "[LEG_ACTION_ABS]", #note: max_std still 0.1
            "TASK.MEASUREMENTS": "[X_LOCATION]",
        },
        "overrides":{
            "RL.SUCCESS_MEASURE": "X_LOCATION",
            "RL.REWARD_MEASURE": "X_LOCATION",
        }
    },
    "ant_move_forward_abs_pos_simple_low_std":{
        #reduce to X_LOCATION only
        "base_experiment":"ant_move_forward_rel_pos_base",
        "task_overrides": {
            #abs position control
            "TASK.POSSIBLE_ACTIONS": "[LEG_ACTION_ABS]", #note: max_std still 0.1
            "TASK.MEASUREMENTS": "[X_LOCATION]",
        },
        "overrides":{
            "RL.SUCCESS_MEASURE": "X_LOCATION",
            "RL.REWARD_MEASURE": "X_LOCATION",
            "RL.POLICY.ACTION_DIST.max_std": "0.04",
        }
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
        overrides += " USE_THREADED_VECTOR_ENV True"

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
        if args.test:
            run_cmd_postfix+=" --test"
        for exp_name in experiments.keys():
            print(run_cmd_prefix + exp_name + run_cmd_postfix)
        print("============================================================")
    else:
        run_type = "eval" if args.type in ["eval", "quick-eval"] else "train"
        run(experiment=args.exp, run_type=run_type, testing=args.test, quick_eval=(args.type == "quick-eval"))