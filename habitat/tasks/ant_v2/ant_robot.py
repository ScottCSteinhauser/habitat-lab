import magnum as mn
import numpy as np

import math

from habitat.tasks.ant_v2.quadruped_wrapper import (
    QuadrupedRobot,
    QuadrupedRobotParams,
    RobotCameraParams,
)


class AntV2Robot(QuadrupedRobot):
    def __init__(self, urdf_path, sim, limit_robo_joints=True, fixed_base=False):
        ant_params = QuadrupedRobotParams(
            hip_joints=[6, 11, 16, 1],
            ankle_joints=[8, 13, 18, 3],

            hip_init_params=[0,0,0,0],
            ankle_init_params=[-1,1,1,-1],

            cameras={
                "robot_arm": RobotCameraParams(
                    cam_offset_pos=mn.Vector3(0, 0.0, 0.1),
                    cam_look_at_pos=mn.Vector3(0.1, 0.0, 0.0),
                    attached_link_id=22,
                    relative_transform=mn.Matrix4.rotation_y(mn.Deg(-90))
                    @ mn.Matrix4.rotation_z(mn.Deg(90)),
                ),
                "robot_head": RobotCameraParams(
                    cam_offset_pos=mn.Vector3(0.17, 1.2, 0.0),
                    cam_look_at_pos=mn.Vector3(0.75, 1.0, 0.0),
                    attached_link_id=-1,
                ),
                "robot_third": RobotCameraParams(
                    cam_offset_pos=mn.Vector3(-10.5, 1.7, -0.5),
                    cam_look_at_pos=mn.Vector3(1, 0.0, 0.75),
                    attached_link_id=-1,
                ),
            },

            hip_mtr_pos_gain=0.1, # reactivity towards target position
            hip_mtr_vel_gain=0.1, # damping measure
            hip_mtr_max_impulse=10.0, # max force aloud in one physics timestep
            ankle_mtr_pos_gain=0.1,
            ankle_mtr_vel_gain=0.1,
            ankle_mtr_max_impulse=10.0,
            # can optimize with fixed base & random target joint positions
            # also should test on ground (ant shouldn't fling itself around)

            base_offset=mn.Vector3(0,0,0),
            base_link_names={
                "torso",
            },
        )
        self.physics_manager = sim.get_rigid_object_manager()
        super().__init__(ant_params, urdf_path, sim, limit_robo_joints, fixed_base)


    def reconfigure(self) -> None:
        super().reconfigure()

        # NOTE: this is necessary to set locked head and back positions
        self.update()

    def reset(self) -> None:
        super().reset()

        # NOTE: this is necessary to set locked head and back positions
        self.update()

    @property
    def base_transformation(self):
        add_rot = mn.Matrix4.rotation(mn.Rad(-np.pi / 2), mn.Vector3(1.0, 0, 0))
        return self.sim_obj.transformation @ add_rot

    def update(self):
        super().update()
        
    def natural_walking_gait_at(self, time, ankle_amplitude, ankle_period_offset, leg_amplitude):
        """Compute a leg state vector for periodic motion at a given time [0,1]."""
        # Simple walking pattern for the ant
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
