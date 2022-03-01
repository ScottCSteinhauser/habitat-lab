#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import defaultdict, OrderedDict
from typing import Any, List, Optional, Tuple, Dict, Union
import random

import attr
import numpy as np
from gym import spaces

import habitat_sim
from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    DepthSensor,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.core.spaces import ActionSpace
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationTask
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.utils.visualizations import fog_of_war, maps

try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
    from habitat_sim import RigidState
    from habitat_sim.physics import VelocityControl
except ImportError:
    pass

try:
    import magnum as mn
except ImportError:
    pass

# import quadruped_wrapper
from habitat.tasks.ant_v2.ant_robot import AntV2Robot
from habitat.tasks.ant_v2.ant_v2_sim_debug_utils import AntV2SimDebugVisualizer


def merge_sim_episode_with_object_config(sim_config, episode):
    sim_config.defrost()
    sim_config.ep_info = [episode.__dict__]
    sim_config.freeze()
    return sim_config


@registry.register_simulator(name="Ant-v2-sim")
class AntV2Sim(HabitatSim):
    def __init__(self, config):
        super().__init__(config)

        agent_config = self.habitat_config
        self.first_setup = True
        self.is_render_obs = False
        self.ep_info = None
        self.prev_loaded_navmesh = None
        self.prev_scene_id = None
        self.enable_physics = True
        self.robot = None
        
        # The leg target state for the ant
        self.leg_target_state = None
        
        if config.LEG_TARGET_STATE == "RANDOM":
            self.leg_target_state = np.random.rand(8) * 2 - 1
        else:
            self.leg_target_state = np.array(config.LEG_TARGET_STATE)
        self.leg_target_state = np.array([float(x) for x in self.leg_target_state])
        
        
        # The direction we want the ant to progress in. The magnitude is also the desired velocity
        self.target_vector = None
        self.target_speed = config.TARGET_SPEED
        self.ant_rotation = config.ANT_START_ROTATION # can be a float or string "Random" indicating a random initialization each time

        if config.TARGET_VECTOR == "RANDOM":
            self.generate_random_target_vector()
        else:
            self.target_vector = np.array(config.TARGET_VECTOR)
        self.target_vector = np.array([float(x) for x in self.target_vector])
        
        
        
        
        #used to measure root transformation delta for reward
        self.prev_robot_transformation = None
        
        # Stores the history of actions taken by the ant, relevant for rewarding smoother actions and giving the ant context about past joint positions
        self.action_history = []
        
        # Stores the history of actions taken by the ant, relevant giving the ant context about past joint positions
        self.joint_position_history = []

        #the control rate in Hz. Simulator is stepped at 1.0/ctrl_freq.
        #NOTE: should be balanced with ENVIRONMENT.MAX_EPISODE_STEPS and RL.PPO.num_steps
        self.ctrl_freq = agent_config.CTRL_FREQ
        self.elapsed_steps = None
        
        self.load_obstacles = False
        # self.load_obstacles = agent_config.LOAD_OBSTACLES # Not working during training!


        self.art_objs = []
        self.start_art_states = {}
        self.cached_art_obj_ids = []
        self.scene_obj_ids = []
        self.viz_obj_ids = []
        # Used to get data from the RL environment class to sensors.
        self.track_markers = []
        self._goal_pos = None
        self.viz_ids: Dict[Any, Any] = defaultdict(lambda: None)
        self.concur_render = self.habitat_config.get(
            "CONCUR_RENDER", True
        ) and hasattr(self, "get_sensor_observations_async_start")
        #print(f"self.concur_render = {self.concur_render}")

        self.debug_visualizer = AntV2SimDebugVisualizer(self)

    def generate_random_target_vector(self):
        x, y = None, None
        while True:
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if math.sqrt(x**2 + y**2) > 1:
                continue
            break
        self.target_vector = np.array([x, 0, y])/np.linalg.norm(np.array([x, 0, y]))

    def _try_acquire_context(self):
        # Is this relevant?
        if self.concur_render:
            self.renderer.acquire_gl_context()

    def reconfigure(self, config):
        ep_info = config["ep_info"][0]
        # ep_info = self._update_config(ep_info)

        config["SCENE"] = ep_info["scene_id"]
        super().reconfigure(config)

        #add eval/debug visualizations if in eval mode
        self.is_eval = "THIRD_SENSOR" in self.habitat_config.AGENT_0.SENSORS
        self.robot_root_path = []
        #print(f"self.is_eval = {self.is_eval}")

        self.ep_info = ep_info

        self.target_obj_ids = []

        self.scene_id = ep_info["scene_id"]

        self._try_acquire_context()

        self.joint_position_history = []
        self.action_history = []          
        
        if self.robot is None: # Load the environment
            # # get the primitive assets attributes manager
            prim_templates_mgr = self.get_asset_template_manager()

            # get the physics object attributes manager
            obj_templates_mgr = self.get_object_template_manager()

            # get the rigid object manager
            rigid_obj_mgr = self.get_rigid_object_manager()

            # add ant
            self.robot = AntV2Robot(self.habitat_config.ROBOT_URDF, self)
            self.robot.reconfigure()
            self.robot.base_pos = mn.Vector3(
                self.habitat_config.AGENT_0.START_POSITION
            )
            if self.ant_rotation == "RANDOM":
                self.robot.base_rot = random.uniform(-1*math.pi, math.pi)
            else:
                self.robot.base_rot = self.ant_rotation
                
            self.prev_robot_transformation = self.robot.base_transformation

            # add floor
            cube_handle = obj_templates_mgr.get_template_handles("cube")[0]
            floor = obj_templates_mgr.get_template_by_handle(cube_handle)
            #should be thicker than 0.08 for better collision margin stability
            floor.scale = np.array([20.0, 0.1, 20.0])

            obj_templates_mgr.register_template(floor, "floor")
            floor_obj = rigid_obj_mgr.add_object_by_template_handle("floor")
            floor_obj.translation = np.array([2.50, -1, 0.5])
            floor_obj.motion_type = habitat_sim.physics.MotionType.STATIC
            
            obstacles = []
            if self.load_obstacles:
                # load periodically placed obstacles
                # add floor
                cube_obstacle = obj_templates_mgr.get_template_by_handle(cube_handle)
                cube_obstacle.scale = np.array([0.1, 1, 4.8])
                # TODO: COLOR OBSTACLE RED
                obj_templates_mgr.register_template(cube_obstacle, "cube_obstacle")
                
                for i in range(6):
                    obstacles.append(rigid_obj_mgr.add_object_by_template_handle("cube_obstacle"))
                    obstacles[-1].motion_type = habitat_sim.physics.MotionType.KINEMATIC
                    obstacles[-1].translation = np.array([i*3 + 2, -0.5, 5 * (1 - 2 * (i % 2))])
                    obstacles[-1].motion_type = habitat_sim.physics.MotionType.STATIC
                    
                    
        else: # environment is already loaded; reset the Ant
            self.robot.reset()
            self.robot.base_pos = mn.Vector3(
                self.habitat_config.AGENT_0.START_POSITION
            )
            self.robot.base_rot = self.ant_rotation
            
            if config.LEG_TARGET_STATE == "RANDOM":
                self.leg_target_state = np.random.rand(8) * 2 - 1
            
            if config.TARGET_VECTOR == "RANDOM":
                self.generate_random_target_vector()
                
            self.prev_robot_transformation = self.robot.base_transformation
        self.elapsed_steps = 0

    def step(self, action):
        # add robot joint position to history
        self.joint_position_history.append(self.robot.leg_joint_state)
        
        #cache the position before updating
        self.step_physics(1.0 / self.ctrl_freq)
        self.prev_robot_transformation = self.robot.base_transformation
        if self.is_eval:
            self.robot_root_path.append(self.robot.base_pos)

        # returns new observation after step
        self._prev_sim_obs = self.get_sensor_observations()
        obs = self._sensor_suite.get_observations(self._prev_sim_obs)
        self.elapsed_steps += 1
        return obs

    def debug_draw(self):
        if self.is_eval:
            #draw some debug line visualizations
            #self.debug_visualizer.draw_axis()
            if len(self.robot_root_path) > 1:
                self.debug_visualizer.draw_path(self.robot_root_path)
            
            # draw ant's orientation vector
            self.debug_visualizer.draw_vector(mn.Vector3(self.robot.base_pos), self.robot.base_transformation.up)
            # draw ant's forward directional vector
            ant_forward_vector = self.robot.base_transformation.transform_vector(mn.Vector3(1, 0, 0))
            self.debug_visualizer.draw_vector(mn.Vector3(self.robot.base_pos), ant_forward_vector)
            # draw ant's target vector
            tv = mn.Vector3(self.target_vector[0], self.target_vector[1], self.target_vector[2])
            self.debug_visualizer.draw_vector(mn.Vector3(self.robot.base_pos), tv, mn.Color4(0.3, 1.0, 0.3, 1.0))

@registry.register_sensor
class AntObservationSpaceSensor(Sensor):

    cls_uuid: str = "ant_observation_space_sensor"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        #compute the size of the observation from active terms
        self._observation_size = 0
        for active_term in config.ACTIVE_TERMS:
            if active_term == "JOINT_POSITION_HISTORY" or active_term == "ACTION_HISTORY" :
                self._observation_size += config.get(active_term).SIZE * config.get(active_term).NUM_STEPS
            else:
                self._observation_size += config.get(active_term).SIZE
        print(self._observation_size)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._observation_size,), dtype=np.float
        )

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.NORMAL

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        obs_terms = []
        
        if "PERIODIC_TIME" in self.config.ACTIVE_TERMS:
            obs_terms.extend([math.fmod(self._sim.get_world_time(), 1.0)])

        if "BASE_POS" in self.config.ACTIVE_TERMS:
            # base position (3D)
            obs_terms.extend([x for x in self._sim.robot.base_pos])

        if "BASE_QUATERNION" in self.config.ACTIVE_TERMS:
            # base orientation (4D) - quaternion
            obs_terms.extend([x for x in list(self._sim.robot.base_rot.vector)])
            obs_terms.extend([self._sim.robot.base_rot.scalar])
            
        if "EGOCENTRIC_TARGET_VECTOR" in self.config.ACTIVE_TERMS:
            # gives the target vector in local space (3D)
            tv = [float(x) for x in self._sim.target_vector]
            egocentric_target_vector = self._sim.robot.base_transformation.inverted().transform_vector(mn.Vector3(tv[0], tv[1], tv[2]))
            obs_terms.extend([x for x in list(egocentric_target_vector)])
            
        if "EGOCENTRIC_UPWARDS_VECTOR" in self.config.ACTIVE_TERMS:
            # gives the global up vector (0,1,0) in local space (3D)
            uv = mn.Vector3(0,1,0)
            egocentric_upwards_vector = self._sim.robot.base_transformation.inverted().transform_vector(uv)
            obs_terms.extend([x for x in list(egocentric_upwards_vector)])

        if "BASE_LIN_VEL" in self.config.ACTIVE_TERMS:
            # base linear velocity (3D)
            obs_terms.extend([x for x in list(self._sim.robot.base_velocity)])
        
        if "EGOCENTRIC_BASE_LIN_VEL" in self.config.ACTIVE_TERMS:
            # base linear velocity (3D)
            bv = self._sim.robot.base_velocity
            egocentric_base_lin_vel = self._sim.robot.base_transformation.inverted().transform_vector(bv)
            obs_terms.extend([x for x in list(egocentric_base_lin_vel)])

        if "BASE_ANG_VEL" in self.config.ACTIVE_TERMS:
            # base angular velocity (3D)
            obs_terms.extend([x for x in list(self._sim.robot.base_angular_velocity)])

        if "EGOCENTRIC_BASE_ANG_VEL" in self.config.ACTIVE_TERMS: # might be not working
            # base linear velocity (3D)
            bv = self._sim.robot.base_angular_velocity
            egocentric_base_ang_vel = self._sim.robot.base_transformation.inverted().transform_vector(bv)
            obs_terms.extend([x for x in list(egocentric_base_ang_vel)])

        if "JOINT_VEL" in self.config.ACTIVE_TERMS:
            # ant joint velocity (8D)
            obs_terms.extend([x for x in list(self._sim.robot.joint_velocities)])

        if "JOINT_POS" in self.config.ACTIVE_TERMS:
            # ant joint position states (8D) (where am I now?)
            # NOTE: this is the state used in joint based rewards
            obs_terms.extend([x for x in list(self._sim.robot.leg_joint_state)])

        if "JOINT_MOTOR_POS" in self.config.ACTIVE_TERMS:
            # ant joint motor targets (8D) (where do I want to be?) (Radians)
            # NOTE: this is the state modified by the action
            obs_terms.extend([x for x in list(self._sim.robot.leg_joint_pos)])

        if "JOINT_TARGET" in self.config.ACTIVE_TERMS:
            # joint state position target (8D)
            obs_terms.extend([x for x in list(self._sim.leg_target_state)])
        
        if "NEXT_JOINT_TARGET" in self.config.ACTIVE_TERMS:
            # joint state position target (8D) (for timestep t+1)
            obs_terms.extend([x for x in list(self._sim.leg_target_state)])
            
        if "JOINT_POSITION_HISTORY" in self.config.ACTIVE_TERMS:
            # joint state position target (8D) 
            # get size of history (Number of previous steps to include)
            # Number of dimensions = 8 * num_steps
            for i in range(-self.config.JOINT_POSITION_HISTORY.NUM_STEPS, 0):
                if (self.config.JOINT_POSITION_HISTORY.NUM_STEPS > len(self._sim.joint_position_history)):
                    obs_terms.extend([0]*8)
                else:
                    obs_terms.extend([x for x in list(self._sim.joint_position_history[i])])
                    #print(self._sim.joint_position_history[i])
        
        if "ACTION_HISTORY" in self.config.ACTIVE_TERMS:
            # Previous actions (8D)
            # get size of history (Number of previous steps to include)
            # Number of dimensions = 8 * num_steps
            for i in range(-self.config.ACTION_HISTORY.NUM_STEPS, 0):
                if (self.config.ACTION_HISTORY.NUM_STEPS > len(self._sim.action_history)):
                    obs_terms.extend([0]*8)
                else:
                    obs_terms.extend([x for x in list(self._sim.action_history[i])])
                    print(self._sim.action_history[i])

        #TODO: add terms for ego centric up(3), forward(3), target_velocity(3)

        return np.array(obs_terms)

@registry.register_task_action
class LegRelPosAction(SimulatorTaskAction):
    """
    The leg motor targets are offset by the delta joint values specified by the
    action
    """

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.LEG_JOINT_DIMENSIONALITY,),
            low=-1,
            high=1,
            dtype=np.float32,
        )

    def step(self, delta_pos, should_step=True, *args, **kwargs):
        # clip from -1 to 1
        delta_pos = np.clip(delta_pos, -1, 1)
        
        # record the normalized action for use in the ActionCost measure
        self._sim.action_history.append(np.copy(delta_pos))
        
        #NOTE: DELTA_POS_LIMIT==1 results in max policy output covering full joint range (-1, 1) radians in 2 timesteps
        delta_pos *= self._config.DELTA_POS_LIMIT
        self._sim: AntV2Sim
        #clip the motor targets to the joint range
        self._sim.robot.leg_joint_pos = np.clip(delta_pos + self._sim.robot.leg_joint_pos, self._sim.robot.joint_limits[0], self._sim.robot.joint_limits[1])
        if should_step:
            return self._sim.step(HabitatSimActions.LEG_VEL)
        
        return None

@registry.register_task_action
class LegAbsPosAction(SimulatorTaskAction):
    """
    The leg motor targets are specified by the
    action
    """

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.LEG_JOINT_DIMENSIONALITY,),
            low=-1,
            high=1,
            dtype=np.float32,
        )

    def step(self, pos, should_step=True, *args, **kwargs):
        # clip from -1 to 1
        pos = np.clip(pos, -1, 1)

        # record the normalized action for use in the ActionCost measure
        self._sim.action_history.append(np.copy(pos))

        self._sim: AntV2Sim
        #clip the motor targets to the joint range
        self._sim.robot.leg_joint_pos = np.clip(pos, self._sim.robot.joint_limits[0], self._sim.robot.joint_limits[1])
        if should_step:
            return self._sim.step(HabitatSimActions.LEG_VEL)

        return None


@registry.register_task_action
class LegRelPosActionSymmetrical(SimulatorTaskAction):
    """
    The leg motor targets are offset by the delta joint values specified by the
    action, symmetry is enforced
    """

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.LEG_JOINT_DIMENSIONALITY,),
            low=-1,
            high=1,
            dtype=np.float32,
        )

    def step(self, action, should_step=True, *args, **kwargs):
        # clip from -1 to 1
        action = np.clip(action, -1, 1) # should be 4 dimensions
        
        # take the 4 dimensions and apply them to both sides of the ant
        delta_pos = []
        delta_pos.extend(action[0:2])
        delta_pos.extend(action[0:2])
        delta_pos[-2] *= -1
        delta_pos.extend(action[2:4])
        delta_pos.extend(action[2:4])
        delta_pos[-2] *= -1
        
        # record the normalized action for use in the ActionCost measure
        self._sim.action_history.append(np.copy(delta_pos))
        
        #NOTE: DELTA_POS_LIMIT==1 results in max policy output covering full joint range (-1, 1) radians in 2 timesteps
        delta_pos *= self._config.DELTA_POS_LIMIT
        
        self._sim: AntV2Sim
        #clip the motor targets to the joint range
        self._sim.robot.leg_joint_pos = np.clip(delta_pos + self._sim.robot.leg_joint_pos, self._sim.robot.joint_limits[0], self._sim.robot.joint_limits[1])
        if should_step:
            return self._sim.step(HabitatSimActions.LEG_VEL)

        return None

@registry.register_task_action
class LegRelPosActionGaitDeviation(SimulatorTaskAction):
    """
    The ant walks in a sunisoidal walking motion by default, action is the deviation in joint positions from that.
    """

    @property
    def action_space(self):
        return spaces.Box(
            shape=(self._config.LEG_JOINT_DIMENSIONALITY,),
            low=-1,
            high=1,
            dtype=np.float32,
        )

    def periodic_leg_motion_at(self, time, ankle_amplitude, ankle_period_offset, leg_amplitude):
        """Compute a leg state vector for periodic motion at a given time [0,1]."""
        # Simple walking pattern for ant
        joint_state = np.zeros(8)
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

    def step(self, delta_pos, should_step=True, *args, **kwargs):
        delta_pos = np.clip(delta_pos, -1, 1)
        
        # record the action for use in the ActionCost measure, after clipping, before shrinking
        self._sim.action_history.append(np.copy(delta_pos))

        #NOTE: DELTA_POS_LIMIT==1 results in max policy output covering full joint range (-1, 1) radians in 2 timesteps
        delta_pos *= self._config.DELTA_POS_LIMIT
        
        self._sim: AntV2Sim
        #clip the motor targets to the joint range
        t = math.fmod(self._sim.get_world_time(), 1.0)

        natural_ant_gait = self.periodic_leg_motion_at(t, 0.23, -0.26, 0.775)
        target_pos = np.clip(natural_ant_gait + delta_pos, -1, 1)
        
        self._sim.robot.leg_joint_pos = np.clip(target_pos, self._sim.robot.joint_limits[0], self._sim.robot.joint_limits[1])
        
        # update target position to be the natural gait
        self._sim.leg_target_state = natural_ant_gait
        
        if should_step:
            return self._sim.step(HabitatSimActions.LEG_VEL)
        
        return None


@registry.register_task_action
class LegAction(SimulatorTaskAction):
    """A continuous leg control into one action space."""

    def __init__(self, *args, config, sim: AntV2Sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        leg_controller_cls = eval(self._config.LEG_CONTROLLER)
        self.leg_ctrlr = leg_controller_cls(
            *args, config=config, sim=sim, **kwargs
        )

    def reset(self, *args, **kwargs):
        self.leg_ctrlr.reset(*args, **kwargs)

    @property
    def action_space(self):
        action_spaces = {
            "leg_action": self.leg_ctrlr.action_space,
        }
        return spaces.Dict(action_spaces)

    def step(self, leg_action, *args, **kwargs):
        self.leg_ctrlr.step(leg_action, should_step=False)
        return self._sim.step(HabitatSimActions.LEG_ACTION)


@registry.register_task(name="Ant-v2-task")
class AntV2Task(NavigationTask):
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)
        
    def overwrite_sim_config(self, sim_config, episode):
        return merge_sim_episode_with_object_config(sim_config, episode)
