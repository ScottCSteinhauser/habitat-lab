#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import defaultdict
from typing import Any, List, Optional, Tuple, Dict

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
        #used to measure root position delta for reward
        self.prev_robot_pos = None
        
        #used to give reward for magnitude of action
        self.most_recent_action = None

        # Number of physics updates per action
        #self.ac_freq_ratio = agent_config.AC_FREQ_RATIO
        # The physics update time step.
        #self.ctrl_freq = agent_config.CTRL_FREQ
        # Effective control speed is (ctrl_freq/ac_freq_ratio)
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

    def _try_acquire_context(self):
        # Is this relevant?
        if self.concur_render:
            self.renderer.acquire_gl_context()

    def reconfigure(self, config):
        ep_info = config["ep_info"][0]
        # ep_info = self._update_config(ep_info)

        config["SCENE"] = ep_info["scene_id"]
        super().reconfigure(config)

        self.ep_info = ep_info

        self.target_obj_ids = []

        self.scene_id = ep_info["scene_id"]

        self._try_acquire_context()

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
            self.robot.base_rot = math.pi / 2
            self.prev_robot_pos = self.robot.base_pos

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
            self.robot.base_rot = math.pi / 2
            self.prev_robot_pos = self.robot.base_pos

    def step(self, action):
        #cache the position before updating
        self.prev_robot_pos = self.robot.base_pos
        self.step_physics(1.0 / 30.0)

        # returns new observation after step
        self._prev_sim_obs = self.get_sensor_observations()
        obs = self._sensor_suite.get_observations(self._prev_sim_obs)
        return obs


@registry.register_sensor
class AntObservationSpaceSensor(Sensor):

    cls_uuid: str = "ant_observation_space_sensor"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(37,), dtype=np.float
        )

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.NORMAL

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        obs = self._sim.robot.observational_space
        return obs

class VirtualMeasure(Measure):
    """Implements some basic functionality to avoid duplication."""

    cls_uuid: str = "VIRTUAL_MEASURE_DO_NOT_USE"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._metric = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = None


@registry.register_measure
class XLocation(VirtualMeasure):
    """The measure calculates the x component of the robot's location."""

    cls_uuid: str = "X_LOCATION"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(sim, config, args)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        if self._metric is None:
            self._metric = None

        current_position = self._sim.robot.base_pos
        self._metric = current_position.x

@registry.register_measure
class VectorRootDelta(VirtualMeasure):
    """Measures the agent's root motion along a target vector."""

    cls_uuid: str = "VECTOR_ROOT_DELTA"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        #NOTE: should be normalized, start with X axis
        self.vector = np.array([1,0,0])
        super().__init__(sim, config, args)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        if self._metric is None or self._sim.prev_robot_pos is None:
            self._metric = None
        #projected_vector = (displacement.dot(v) / norm(v)^2) * v
        #v is unit, so magnitude reduces to displacement.dot(v)
        displacement = self._sim.robot.base_pos - self._sim.prev_robot_pos
        self._metric = np.dot(displacement, self.vector)

@registry.register_measure
class JointStateError(VirtualMeasure):
    """The measure calculates the error between current and target joint states"""

    cls_uuid: str = "JOINT_STATE_ERROR"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        #TODO: dynamic targets, for now just a rest pose
        self.target_state = np.array([0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0])
        super().__init__(sim, config, args)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        if self._metric is None:
            self._metric = None

        current_state = self._sim.robot.leg_joint_state
        
        self._metric = -np.linalg.norm(np.abs(current_state - self.target_state))
        #print(self._metric)

@registry.register_measure
class ActiveContacts(VirtualMeasure):
    # TODO: Set this such that only contact points made by the ant are considered.
    # Inspired by MuJoCo's ant-v2 reward structure
    """The measure calculates the number of active contact points in the environment"""

    cls_uuid: str = "ACTIVE_CONTACTS"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(sim, config, args)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        if self._metric is None:
            self._metric = None
        
        self._metric = self._sim.get_physics_num_active_contact_points()
        #print(self._metric)

@registry.register_measure
class ActionCost(VirtualMeasure):
    # TODO: Set this such that only contact points made by the ant are considered.
    # Inspired by MuJoCo's ant-v2 reward structure
    """Actions which have greater magnitudes are more costly."""

    cls_uuid: str = "ACTION_COST"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(sim, config, args)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        if self._metric is None:
            self._metric = None
        
        self._metric = np.sum(np.square(self._sim.most_recent_action))
        #print(self._metric)

@registry.register_measure
class CompositeAntReward(VirtualMeasure):
    """The measure calculates the error between current and target joint states"""

    cls_uuid: str = "COMPOSITE_ANT_REWARD"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        #add all potential dependencies here:
        self.measure_dependencies=[
            VectorRootDelta.cls_uuid,
            # JointStateError.cls_uuid,
            ActiveContacts.cls_uuid,
            ActionCost.cls_uuid,
        ]

        #NOTE: define active rewards and weights here:
        self.active_measure_weights = {
            VectorRootDelta.cls_uuid: 10.0,
            # JointStateError.cls_uuid: 1.0,
            ActiveContacts.cls_uuid: -0.05,
            ActionCost.cls_uuid: -0.5,
        }
        
        super().__init__(sim, config, args)

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            self.measure_dependencies,
        )

        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        reward = 0
        #weight and combine reward terms
        for measure_uuid,weight in self.active_measure_weights.items():
            measure = task.measurements.measures[measure_uuid]
            if measure.get_metric():
                reward += measure.get_metric()*weight
            #debugging: measure metrics will be None upon init
            #else:
            #    print(f"warning, {measure_uuid} is None")

        self._metric = reward

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
        #NOTE: DELTA_POS_LIMIT==1 results in max policy output covering full joint range (-1, 1) radians in 2 timesteps
        delta_pos *= self._config.DELTA_POS_LIMIT
        self._sim: AntV2Sim
        #clip the motor targets to the joint range
        self._sim.robot.leg_joint_pos = np.clip(delta_pos + self._sim.robot.leg_joint_pos, self._sim.robot.joint_limits[0], self._sim.robot.joint_limits[1])
        if should_step:
            return self._sim.step(HabitatSimActions.LEG_VEL)
        
        # record the action for use in the ActionCost measure
        self._sim.most_recent_action = delta_pos
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
        
        # left off trying to get action so I can compute magnitude and use this as a metric.
    """def step(self, action: Dict[str, Any], episode: Episode):
        super().step(action=action, episode=episode)
        print(action)
        #self._sim.most_recent_action = action
        print(self._sim.most_recent_action)"""
        
    def overwrite_sim_config(self, sim_config, episode):
        return merge_sim_episode_with_object_config(sim_config, episode)
