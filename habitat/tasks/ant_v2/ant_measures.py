#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from gym import spaces

from typing import Any, List, Optional, Tuple, Dict, Union

from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.ant_v2.ant_v2 import AntV2Sim

from habitat.core.simulator import (
    AgentState,
    DepthSensor,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)


try:
    import magnum as mn
except ImportError:
    pass

from habitat.config import Config


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
        super().__init__(sim, config, args)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        self.vector = self._sim.target_vector
        if self._metric is None or self._sim.prev_robot_transformation is None:
            self._metric = None
        #projected_vector = (displacement.dot(v) / norm(v)^2) * v
        #v is unit, so magnitude reduces to displacement.dot(v)
        displacement = self._sim.robot.base_pos - self._sim.prev_robot_transformation.translation
        self._metric = np.dot(displacement, self.vector)


@registry.register_measure
class VelocityAlignment(VirtualMeasure):
    """Measures the agent's root velocity alignment along a target vector."""

    cls_uuid: str = "VELOCITY_ALIGNMENT"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        #NOTE: should be normalized, start with X axis
        super().__init__(sim, config, args)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        self.vector = self._sim.target_vector
        if self._metric is None or self._sim.prev_robot_transformation is None:
            self._metric = None
        ant_velocity = self._sim.robot.base_velocity
        ant_velocity_unit_vector = ant_velocity / np.linalg.norm(ant_velocity)
        
        # we have two normalized vectors, metric is just the dot product
        alignment = np.dot(ant_velocity_unit_vector, self.vector)
        self._metric = alignment
        # print("vel alignment:", self._metric)

@registry.register_measure
class OrthogonalVelocity(VirtualMeasure):
    """Measures the agent's root velocity alignment orthogonal a target vector."""

    cls_uuid: str = "ORTHOGONAL_VELOCITY"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        #NOTE: should be normalized, start with X axis
        super().__init__(sim, config, args)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        self.target_vector = self._sim.target_vector
        if self._metric is None or self._sim.prev_robot_transformation is None:
            self._metric = None

        ant_velocity = self._sim.robot.base_velocity
        orthogonal_velocity = ant_velocity - np.dot(ant_velocity, self.target_vector) * self.target_vector
        
        self._metric = -np.linalg.norm(orthogonal_velocity)
        #print("orthogonal component:", self._metric)

@registry.register_measure
class SpeedTarget(VirtualMeasure):
    """Measures the agent's velocity in the target vector's direction."""

    cls_uuid: str = "SPEED_TARGET"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        #NOTE: should be normalized, start with X axis
        super().__init__(sim, config, args)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        self.target_vector = self._sim.target_vector
        self.target_speed = self._sim.target_speed
        if self._metric is None or self._sim.prev_robot_transformation is None:
            self._metric = None
        ant_velocity = self._sim.robot.base_velocity
        # target vector should be normalized
        ant_velocity_in_target_direction = np.dot(ant_velocity, self.target_vector)
        
        # we have two normalized vectors, metric is just the dot product
        similarity_score = 1 - abs(ant_velocity_in_target_direction / self.target_speed - 1)
        self._metric = similarity_score
        # print("speed:", ant_velocity_in_target_direction, self._metric)

@registry.register_measure
class JointStateError(VirtualMeasure):
    """The measure calculates the error between current and target joint states"""

    cls_uuid: str = "JOINT_STATE_ERROR"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._normalized = False if not config.NORMALIZED else config.NORMALIZED
       #NOTE: computed first time (scalar)
        self.joint_norm_scale = None
        super().__init__(sim, config, args)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        self.target_state = self._sim.leg_target_state # np.array([0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0])
 
        if self._metric is None:
            self._metric = None

        current_state = self._sim.robot.leg_joint_state

        if self._normalized:
            if self.joint_norm_scale is None:
                lims = self._sim.robot.joint_limits
                #per-element maximum error between lower|upper limits and target
                max_errors = np.fmax(np.abs(self.target_state-lims[0]), np.abs(self.target_state-lims[1]))
                self.joint_norm_scale = np.linalg.norm(max_errors)
            self._metric = -np.linalg.norm(current_state - self.target_state)/self.joint_norm_scale
        else:
            self._metric = -np.linalg.norm(current_state - self.target_state)
        # print(self._metric)

@registry.register_measure
class JointStateProductError(VirtualMeasure):
    """The measure calculates the error between current and target joint states as a product of normalized terms [0,1]"""

    cls_uuid: str = "JOINT_STATE_PRODUCT_ERROR"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        #NOTE: computed first time
        self.joint_norm_scale = None
        super().__init__(sim, config, args)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        #TODO: dynamic targets, for now just a rest pose
        self.target_state = self._sim.leg_target_state 
        if self._metric is None:
            self._metric = None
        if self.joint_norm_scale is None:
            lims = self._sim.robot.joint_limits
            #per-element maximum error between lower|upper limits and target
            max_errors = np.fmax(np.abs(self.target_state-lims[0]), np.abs(self.target_state-lims[1]))
            self.joint_norm_scale = np.reciprocal(max_errors)
        current_state = self._sim.robot.leg_joint_state
        #print(f"current_state = {current_state}")
        normalized_errors = np.abs(current_state-self.target_state)*self.joint_norm_scale
        #print(f"self.joint_norm_scale = {self.joint_norm_scale}")
        #print(f"normalized_errors = {normalized_errors}")
        self._metric = np.prod(np.ones(len(self.target_state)) - normalized_errors)
        #print(self._metric)

@registry.register_measure
class VectorAlignmentValue(VirtualMeasure):
    """The measure takes the dot product of the a vector in the ant's local space and a global vector."""
    cls_uuid: str = "VECTOR_ALIGNMENT_VALUE"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self.config = config
        if self.config.UUID:
            self.cls_uuid = self.config.UUID
        
        super().__init__(sim, config, args)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        #TODO: dynamic targets, for now just a rest pose
        self.local_vector = np.array(self.config.LOCAL_VECTOR)
        if self.config.GLOBAL_VECTOR == "TARGET":
            self.global_vector = self._sim.target_vector
        else:
            self.global_vector = np.array(self.config.GLOBAL_VECTOR)
        self.local_vector = np.array([float(x) for x in self.local_vector])
        self.global_vector = np.array([float(x) for x in self.global_vector])
        
        if self._metric is None:
            self._metric = None
            
        globalized_local_vector = self._sim.robot.base_transformation.transform_vector(mn.Vector3(self.local_vector[0], self.local_vector[1], self.local_vector[2]))
        alignment = np.dot(self.global_vector, globalized_local_vector)
        
        if self.config.MODIFIER == "SQUARED":
            self._metric = alignment ** 2 * (alignment / abs(alignment)) # Square the measure & multiple it by it's sign
        else:
            self._metric = alignment
        # print("vector_alignment:",self._metric)

@registry.register_measure
class JointStateMaxError(VirtualMeasure):
    """The measure calculates the max error between current and target joint states"""

    cls_uuid: str = "JOINT_STATE_MAX_ERROR"

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
        
        self._metric = -np.max(np.abs(current_state - self.target_state))

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
        contact_points = self._sim.get_physics_contact_points()
        total_force = 0
        for contact in contact_points:
            total_force += contact.normal_force
                
        self._metric = total_force
        #print(self._metric)

@registry.register_measure
class ActionCost(VirtualMeasure):
    # Inspired by MuJoCo's ant-v2 reward structure
    """Actions which have greater magnitudes are more costly."""

    cls_uuid: str = "ACTION_COST"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self.config = config
        if self.config.UUID:
            self.cls_uuid = self.config.UUID
        super().__init__(sim, config, args)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        if self._metric is None:
            self._metric = None
        total_reward = 1
        # action_history[-1] values are between -1 and 1, 0 is low cost action
        if len(self._sim.action_history):
            if self.config.MODIFIER == "NORMALIZED_PRODUCT":
                for x in self._sim.action_history[-1]:
                    total_reward *= 1 - abs(x)
            elif self.config.MODIFIER == "NORMALIZED_SUM":
                total_reward -= np.average(np.absolute(self._sim.action_history[-1]))
        else:
            total_reward = 0

        self._metric = total_reward - 1
        #if len(self._sim.action_history):
        #    print(self._sim.action_history[-1], self._metric)

@registry.register_measure
class ActionSmoothness(VirtualMeasure):
    # Inspired by MuJoCo's ant-v2 reward structure
    """Actions more like previous actions are better. Window size is dependent on ACTION_HISTORY"""

    cls_uuid: str = "ACTION_SMOOTHNESS"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(sim, config, args)
        self.window = config.WINDOW

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        if self._metric is None:
            self._metric = None
        total_reward = 1
        
        # Get average of n previous episodes
        #TODO: This assumes the action space dimension is 8
        avg_action = np.zeros(8)
        for i in range(-self.window - 1, -1):
            if abs(i) <= len(self._sim.action_history):
                avg_action += self._sim.action_history[i] / self.window

        # Now get difference between average action & current action normalize
        previous_action = self._sim.action_history[-1]
        
        if len(self._sim.action_history):
            for i in range(len(previous_action)):
                total_reward *= 1-abs(previous_action[i] - avg_action[i]) / 2
        else:
            total_reward = 0
        #magnitude of the action vector rather than sum of squares.
        self._metric = total_reward
        #self._metric = -np.sum(np.square(self._sim.most_recent_action))
        #print("Smoothness",self._metric)

@registry.register_measure
class CompositeAntReward(VirtualMeasure):
    """The measure calculates the error between current and target joint states"""

    cls_uuid: str = "COMPOSITE_ANT_REWARD"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        #assert that the reward term is properly configured
        assert config.COMPONENTS
        assert config.WEIGHTS
        assert len(config.COMPONENTS) == len(config.WEIGHTS)

        #add all potential dependencies from config here:
        self.measure_dependencies=config.COMPONENTS

        #NOTE: define active rewards and weights from config here:
        self.active_measure_weights = {}
        for i,measure_uuid in enumerate(self.measure_dependencies):
            self.active_measure_weights[measure_uuid] = float(config.WEIGHTS[i])
        
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
