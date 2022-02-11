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
    #     "overrides": "",
    # }
    # "example_name": {
    #     "description": "Example: describe your experiment for later.",
    #     "config": "Example: habitat_baselines/config/ant_v2/ppo_ant_v2_eval.yaml",
    #     "overrides": "Example: CHECKPOINT_FOLDER 'data/new_checkpoints_video_enabled_3'"
    # },
    "x_loc_ant": {
        "description": "Reward cumulative progress in X direction.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "overrides": " RL.REWARD_MEASURE X_LOCATION",
    },
    "delta_x_loc_ant": {
        "description": "Reward delta root progress in X direction.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "overrides": " RL.REWARD_MEASURE VECTOR_ROOT_DELTA",
    },
    "joint_error_ant":{
        "description": "Linear penalty for target joint angle error.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "overrides": " RL.REWARD_MEASURE JOINT_STATE_ERROR",
    },
    "joint_max_error_ant":{
        "description": "Linear penalty for target joint angle error.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "overrides": " RL.REWARD_MEASURE JOINT_STATE_MAX_ERROR",
    },
    "composite":{
        "description": "Composite reward term.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "overrides": " RL.REWARD_MEASURE COMPOSITE_ANT_REWARD",
    },
    "composite_joint_regression_action_cost":{
        "description": "Composite reward term.",
        "config": "habitat_baselines/config/ant_v2/ppo_ant_v2_train.yaml",
        "overrides": " RL.REWARD_MEASURE COMPOSITE_ANT_REWARD",
    },
}


run_types = ["eval", "train"]
run_base = "python -u habitat_baselines/run.py"


def run(experiment=None, run_type="train"):
    assert experiment in experiments
    assert run_type in run_types

    exp_info = experiments[experiment]

    full_command = run_base
    
    #add the config
    full_command += " --exp-config " + exp_info["config"]
    #add the type
    full_command += " --run-type " + run_type

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
        #NOTE: number of videos/data_points you want
        overrides += " NUM_ENVIRONMENTS 3"

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

    args = parser.parse_args()
    run(experiment=args.exp, run_type=args.type)