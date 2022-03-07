import os
import shutil
import git
import math

import numpy as np
import cv2
import random

import magnum as mn

import habitat
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

from habitat_sim.utils import viz_utils as vut


repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "../habitat-sim/data")

def joint_space_action_oracle(target, state):
    """Given a target joint state and the current motor state, compute the optimal action."""
    #use this as an orcale to test the action magnitude vs. learned solution
    diff = target-state
    np.clip(diff, -1, 1)
    return diff

def periodic_leg_motion_at(time, ankle_amplitude, ankle_period_offset, leg_amplitude):
    """Compute a leg state vector for periodic motion at a given time [0,1]."""
    hip_ixs = [0,2,4,6]
    ankle_ixs_1 = [1,3,]
    ankle_ixs_2 = [5,7]
    joint_state = np.zeros(8)
    for hip_ix in hip_ixs:
        joint_state[hip_ix] = math.sin(2*math.pi*time)
    for ankle_ix in ankle_ixs_1:
        joint_state[ankle_ix] = 0
    for ankle_ix in ankle_ixs_2:
        joint_state[ankle_ix] = 0
    # Simple swim-like walking pattern for ant
    joint_state[0] = leg_amplitude * (math.sin(math.pi + 2*math.pi*time))
    joint_state[2] = leg_amplitude * (math.sin(math.pi + 2*math.pi*time))
    joint_state[4] = leg_amplitude * (math.sin(2*math.pi*time))
    joint_state[6] = leg_amplitude * (math.sin(2*math.pi*time))
    
    ad1 = (1-ankle_amplitude)
    
    joint_state[1] = -(ankle_amplitude * (math.sin(math.pi/2 + ankle_period_offset*math.pi + 2*math.pi*time)) + ad1)
    joint_state[3] = -(ankle_amplitude * (math.sin(-math.pi/2 + ankle_period_offset*math.pi + 2*math.pi*time)) + ad1)
    joint_state[5] = ankle_amplitude * (math.sin(math.pi/2 + ankle_period_offset*math.pi + 2*math.pi*time)) + ad1
    joint_state[7] = ankle_amplitude * (math.sin(-math.pi/2 + ankle_period_offset*math.pi + 2*math.pi*time)) + ad1
    return joint_state


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
        for i in range(100):
            env.reset()
            step = 0
            
            while not step > 1:
                step+=1
                #print(env._sim.robot.sim_obj.joint_positions)
                
                #sample random action for testing:
                action = env.action_space.sample()
                #override actions for testing:
                #action['action_args']['leg_action'] = np.ones(8)*-1
                #action['action_args']['leg_action'] = np.zeros(8)
                
                #joint_target = np.ones(8)*0.5
                #joint_target = np.array([0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0])
                #joint_target = env._sim.robot.random_pose()
                #joint_target = env._sim.robot.leg_joint_state + 0.2*env._sim.robot.random_pose()
                #joint_target = periodic_leg_motion_at(math.fmod(env._sim.get_world_time(), 1.0), 0.23, -0.26, 0.775)
                #print("target",joint_target)
                set_action = np.array([1]*8) * 0.2 * ((step%2) * 2 - 1)
                action['action_args']['leg_action'] = set_action # joint_space_action_oracle(joint_target, env._sim.robot.leg_joint_pos)
                #print("action", action['action_args']['leg_action'])
                #for i in range(2):
                #    action = env.action_space.sample()
                obs = env.step(action)
                #print(obs['ant_observation_space_sensor'])
                observations.append(obs)
                # keystroke = cv2.waitKey(0)
                
                #print(f"observational_space = {env._sim.observational_space}")

                #NOTE: you can check metrics here:
                #measure_query = "VECTOR_ROOT_DELTA"
                #print(f"{measure_query} = {env.task.measurements.measures[measure_query].get_metric()}")

                #if keystroke == 27:
                #    break
                #print(obs["ant_observation_space_sensor"])
                # cv2.imshow("RGB", obs["robot_third_rgb"])
    
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

