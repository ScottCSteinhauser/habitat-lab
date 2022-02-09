import os
import shutil
import git

import numpy as np
import cv2
import random

import habitat
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

from habitat_sim.utils import viz_utils as vut


repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "../habitat-sim/data")

def ant_environment_example():
    config = habitat.get_config(config_paths="configs/tasks/ant-v2.yaml")
    config.defrost()
    config.TASK.SEED = 5435435643
    config.freeze()
    #random.seed(config.TASK.SEED)
    #np.random.seed(config.TASK.SEED)

    observations = []
    with habitat.Env(config=config) as env:
        env.reset()
        while not env.episode_over:
            print(env._sim.robot.sim_obj.joint_positions)

            #sample random action for testing:
            action = env.action_space.sample()
            #override actions for testing:
            #action['action_args']['leg_action'] = np.ones(8)*-1
            #action['action_args']['leg_action'] = np.zeros(8)

            if env._elapsed_steps == 120:
                env._sim.robot.reset()

            #for i in range(2):
            #    action = env.action_space.sample()
            obs = env.step(action)
            observations.append(obs)
            #keystroke = cv2.waitKey(0)

            #if keystroke == 27:
            #    break
            #print(obs["ant_observation_space_sensor"])
            #cv2.imshow("RGB", obs["robot_third_rgb"])

        env.reset()
    vut.make_video(
        observations,
        "robot_third",
        "color",
        "test_ant_wrapper",
        open_vid=True,
        fps=30,
    )

def main():
    ant_environment_example()


if __name__ == "__main__":
    main()

