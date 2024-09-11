try:
    import pybullet as p
except ImportError:
    raise Exception("Please make sure pybullet is installed. You can install it using pip: pip install pybullet")

# import rospkg
import numpy as np
import mujoco

from .base_controller import BaseController
from .linear_interpolator import LinearInterpolator

from utils.transform_utils import (
    mat2quat, 
    mat2pose,
    quat2mat,
    quat_inverse,   
    quat_multiply,  
    quat2euler,
    pose2mat,
    pose_inv,
    pose_in_A_to_pose_in_B,
    axisangle2quat,
    clip_rotation_wxyz,
    clip_translation,
    quat_distance,
    quat2axisangle
)
from utils.controller_utils import (
    nullspace_torques, 
    opspace_matrices,
    set_goal_position,
    set_goal_orientation,
    orientation_error
)

# Global var for linking pybullet server to multiple ik controller instances if necessary
pybullet_server = None

def reset_controllers():
    """
    Global function for doing one-time clears and restarting of any global controller-related
    specifics before re-initializing each individual controller again
    """
    global pybullet_server
    # Disconnect and reconnect to pybullet server if it exists
    if pybullet_server is not None:
        pybullet_server.disconnect()
        pybullet_server.connect()


def get_pybullet_server():
    """
    Getter to return reference to pybullet server module variable

    Returns:
        PyBulletServer: Server instance running PyBullet
    """
    global pybullet_server
    return pybullet_server

class PyBulletServer():
    """
    Helper class to encapsulate an alias for a single PyBullet server
    """

    def __init__(self):
        # Attributes
        self.server_id = None
        self.is_active = False

        # Bodies: Dict of <bullet_robot_id: robot_name> active in PyBullet simulation
        self.bodies = {}

        # Automatically setup this PyBullet server
        self.connect()

    def connect(self):
        """
        Global function to (re-)connect to pybullet server instance if it's not currently active
        """
        if not self.is_active:
            self.server_id = p.connect(p.DIRECT)

            # Reset simulation (Assumes pre-existing connection to the PyBullet simulator)
            p.resetSimulation(physicsClientId=self.server_id)
            self.is_active = True

    def disconnect(self):
        """
        Function to disconnect and shut down this pybullet server instance.

        Should be called externally before resetting / instantiating a new controller
        """
        if self.is_active:
            p.disconnect(physicsClientId=self.server_id)
            self.bodies = {}
            self.is_active = False

class InverseKinematicsController(BaseController):
    def __init__(
            self,
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
            eef_rot_offset,     # Quat (w,x,y,z) offset to convert mujoco eef to pybullet eef
            qpos_limits,
            control_dim=6,
            bullet_server_id=0,
            control_freq=20,
            policy_freq=10,
            damping_ratio=1e-4,
            ramp_ratio=0.5,
            load_urdf=True,
            ik_pos_limit=None,
            ik_ori_limit=None,
            control_delta = True,
            converge_steps=5,
            **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms
    ):
                
        # Run superclass inits
        super().__init__(sim, eef_name, joint_indexes, actuator_range, ramp_ratio, control_freq, policy_freq)

        self.robot_name = "XArm6"

        self._control_dim = control_dim

        # Determine whether we want to use delta or absolute values as inputs
        self._use_delta = control_delta

        # Rotation offsets (for mujoco eef -> pybullet eef) and rest poses
        self.eef_rot_offset = eef_rot_offset
        self.rotation_offset = None
        self.rest_poses = None

        # Set the reference robot target pos / orientation (to prevent drift / weird ik numerical behavior over time)
        self.reference_target_pos = self.ee_pos
        self.reference_target_orn = mat2quat(self.ee_ori_mat)

        # Bullet server id
        self.bullet_server_id = bullet_server_id

        # Interpolator
        self.interpolator_pos = LinearInterpolator(3, control_freq, policy_freq, ramp_ratio)
        self.interpolator_ori = LinearInterpolator(4, control_freq, policy_freq, ramp_ratio, ori_interpolate="quat")

        # Interpolator-related attributes
        self.ori_ref = None
        self.relative_ori = None

        # Values for initializing pybullet env
        self.ik_robot = None
        self.robot_urdf = None
        self.num_bullet_joints = None
        self.bullet_ee_idx = None
        self.bullet_joint_indexes = None  # Useful for splitting right and left hand indexes when controlling bimanual
        self.ik_command_indexes = None  # Relevant indices from ik loop; useful for splitting bimanual left / right
        self.base_orn_offset_inv = None  # inverse orientation offset from pybullet base to world
        self.converge_steps = converge_steps

        # Set ik limits and override internal min / max
        self.ik_pos_limit = ik_pos_limit
        self.ik_ori_limit = ik_ori_limit

        # Commanded pos and resulting commanded vel
        self.commanded_joint_positions = None
        self.commanded_joint_velocities = None

        # Should be in (0, 1], smaller values mean less sensitivity.
        self.user_sensitivity = 1.0

        # Gains
        self._kp = damping_ratio * np.eye(control_dim)

        # Setup inverse kinematics
        self.setup_inverse_kinematics(load_urdf)

        # Lastly, sync pybullet state to mujoco state
        self.sync_state()

    def setup_inverse_kinematics(self, load_urdf=True):
        """
        This function is responsible for doing any setup for inverse kinematics.

        Inverse Kinematics maps end effector (EEF) poses to joint angles that are necessary to achieve those poses.

        Args:
            load_urdf (bool): specifies whether the robot urdf should be loaded into the sim. Useful flag that
                should be cleared in the case of multi-armed robots which might have multiple IK controller instances
                but should all reference the same (single) robot urdf within the bullet sim

        Raises:
            ValueError: [Invalid eef id]
        """
        # Check if pybullet server is active
        global pybullet_server
        if pybullet_server is None:
            pybullet_server = PyBulletServer()

        # get paths to urdfs
        # pkg_path = rospkg.RosPack().get_path('xarm_description')
        # self.robot_urdf = pkg_path + "/urdf/xarm6_ft_sensor_gripper.urdf"
        self.robot_urdf = "models/robots/xarm6/xarm6_with_ft_sensor_gripper.urdf"

        if load_urdf:
            self.ik_robot = p.loadURDF(fileName=self.robot_urdf, useFixedBase=1, physicsClientId=self.bullet_server_id)
            # Add this to the pybullet server
            get_pybullet_server().bodies[self.ik_robot] = self.robot_name
        else:
            # We'll simply assume the most recent robot (robot with highest pybullet id) is the relevant robot and
            # mark this controller as belonging to that robot body
            self.ik_robot = max(get_pybullet_server().bodies)

        # load the number of joints from the bullet data
        self.num_bullet_joints = p.getNumJoints(self.ik_robot, physicsClientId=self.bullet_server_id)

        # Disable collisions between all the joints
        for joint in range(self.num_bullet_joints):
            p.setCollisionFilterGroupMask(
                bodyUniqueId=self.ik_robot,
                linkIndexA=joint,
                collisionFilterGroup=0,
                collisionFilterMask=0,
                physicsClientId=self.bullet_server_id,
            )

        # Default assumes pybullet has same number of joints compared to mujoco sim - THIS IS NOT THE CASE FOR XARM6
        self.bullet_ee_idx = 16  # link_eef2link_tcp_fixed_offset (aka joint_tcp)
        self.bullet_joint_indexes = [1,2,3,4,5,6]
        self.ik_command_indexes = [0,1,2,3,4,5]

        # Set rotation offsets (for mujoco eef -> pybullet eef) and rest poses
        self.rest_poses = list(self.initial_joint_pos)
        eef_offset = np.eye(4)
        # convert self.eef_rot_offset to xyzw for pybullet
        eef_rot_offset_xyzw = [self.eef_rot_offset[1], self.eef_rot_offset[2], self.eef_rot_offset[3], self.eef_rot_offset[0]]
        eef_offset[:3, :3] = quat2mat(quat_inverse(eef_rot_offset_xyzw))

        self.rotation_offset = eef_offset

        # Simulation will update as fast as it can in real time, instead of waiting for
        # step commands like in the non-realtime case.
        p.setRealTimeSimulation(1, physicsClientId=self.bullet_server_id)

    def sync_state(self):
        """
        Syncs the internal Pybullet robot state to the joint positions of the
        robot being controlled.
        """

        # update model (force update)
        self.update(force=True)

        # sync IK robot state to the current robot joint positions
        self.sync_ik_robot()

    def sync_ik_robot(self, joint_positions=None, sync_last=True):
        """
        Force the internal robot model to match the provided joint angles.

        Args:
            joint_positions (Iterable): Array of joint positions. Default automatically updates to
                current mujoco joint pos state
            sync_last (bool): If False, don't sync the last joint angle. This
                is useful for directly controlling the roll at the end effector.
        """
        if joint_positions is None:
            joint_positions = self.joint_pos
        # else:
        #     # IK sync here
        #     print("joint_positions from IK:", joint_positions)
        #     input("sync")
        #     pass

        num_joints = self.joint_dim

        if not sync_last:
            num_joints -= 1
        
        for i, joint_index in enumerate(self.bullet_joint_indexes):
                p.resetJointState(
                    bodyUniqueId=self.ik_robot,
                    jointIndex=joint_index,
                    targetValue=joint_positions[i],
                    targetVelocity=0,
                    physicsClientId=self.bullet_server_id,
                )                

    def ik_robot_eef_joint_cartesian_pose(self):
        """
        Calculates the current cartesian pose of the last joint of the ik robot with respect to the base frame as
        a (pos, orn) tuple where orn is a wxyz quaternion

        Returns:
            2-tuple:

                - (np.array) position
                - (np.array) orientation
        """
        eef_pos_in_world = np.array(
            p.getLinkState(self.ik_robot, self.bullet_ee_idx, physicsClientId=self.bullet_server_id)[0]
        )
        eef_orn_in_world = np.array(
            p.getLinkState(self.ik_robot, self.bullet_ee_idx, physicsClientId=self.bullet_server_id)[1]
        )
        eef_orn_in_world_wxyz = [eef_orn_in_world[3], eef_orn_in_world[0], eef_orn_in_world[1], eef_orn_in_world[2]]
        eef_pose_in_world = pose2mat((eef_pos_in_world, eef_orn_in_world_wxyz))

        base_pos_in_world = np.array(
            p.getBasePositionAndOrientation(self.ik_robot, physicsClientId=self.bullet_server_id)[0]
        )
        base_orn_in_world = np.array(
            p.getBasePositionAndOrientation(self.ik_robot, physicsClientId=self.bullet_server_id)[1]
        )
        base_orn_in_world_wxyz = [base_orn_in_world[3], base_orn_in_world[0], base_orn_in_world[1], base_orn_in_world[2]]
        base_pose_in_world = pose2mat((base_pos_in_world, base_orn_in_world_wxyz))
        world_pose_in_base = pose_inv(base_pose_in_world)

        # Update reference to inverse orientation offset from pybullet base frame to world frame
        self.base_orn_offset_inv = quat2mat(quat_inverse(base_orn_in_world))

        # Update reference target orientation
        self.reference_target_orn = quat_multiply(self.reference_target_orn, base_orn_in_world)

        eef_pose_in_base = pose_in_A_to_pose_in_B(pose_A=eef_pose_in_world, pose_A_in_B=world_pose_in_base)

        return mat2pose(eef_pose_in_base)
    
    def get_control(self, pos=None, rotation=None, update_targets=False):
        """
        Returns joint positions to control the robot after the target end effector
        position and orientation are updated from arguments @dpos and @rotation.
        If no arguments are provided, joint positions will be computed based
        on the previously recorded target.

        Args:
            dpos (np.array): a 3 dimensional array corresponding to the desired
                change in x, y, and z end effector position.
            rotation (np.array): a rotation matrix of shape (3, 3) corresponding
                to the desired rotation from the current orientation of the end effector.
            update_targets (bool): whether to update ik target pos / ori attributes or not

        Returns:
            np.array: a flat array of joint position commands to apply to try and achieve the desired input control.
        """

        # Sync PyBulletjoint positions for IK.
        self.sync_ik_robot()

        # print("[get_control] abs pos:", pos)
        # print("[get_control] abs ori:", mat2quat(rotation))

        # Compute new target joint positions if arguments are provided
        if (pos is not None) and (rotation is not None):
            self.commanded_joint_positions = np.array(
                self.joint_positions_for_eef_command(pos, rotation, update_targets)
            )
        
        # Absolute joint positions
        positions = self.commanded_joint_positions

        return positions
    
    def inverse_kinematics(self, target_position, target_orientation):
        """
        Helper function to do inverse kinematics for a given target position and
        orientation in the PyBullet world frame.

        Args:
            target_position (3-tuple): desired position
            target_orientation (4-tuple): desired orientation quaternion in wxyz format

        Returns:
            list: list of size @num_joints corresponding to the joint angle solution.
        """
        joint_qpos_limits = self._sim.get_joint_qpos_limits([self._joint_index])[0]
        lower_limits = joint_qpos_limits[:, 0]
        upper_limits = joint_qpos_limits[:, 1]
        # print("[IK] target_position:", target_position)
        # print("[IK] target_orientation:", target_orientation)

        # convert target orientation into xyzw format for pybullet
        target_orientation = [target_orientation[1], target_orientation[2], target_orientation[3], target_orientation[0]]
        ik_solution = list(
            p.calculateInverseKinematics(
                bodyUniqueId=self.ik_robot,
                endEffectorLinkIndex=self.bullet_ee_idx,
                targetPosition=target_position,
                targetOrientation=target_orientation,
                lowerLimits=list(lower_limits),
                upperLimits=list(upper_limits),
                jointRanges=list(upper_limits - lower_limits),
                restPoses=self.rest_poses,
                # jointDamping=[0.1] * self.num_bullet_joints,
                physicsClientId=self.bullet_server_id,
            )
        )
        return list(np.array(ik_solution)[self.ik_command_indexes])

    def joint_positions_for_eef_command(self, pos, rotation, update_targets=False):
        """
        This function runs inverse kinematics to back out target joint positions
        from the provided end effector command.

        Args:
            dpos (np.array): a 3 dimensional array corresponding to the desired
                change in x, y, and z end effector position.
            rotation (np.array): a rotation matrix of shape (3, 3) corresponding
                to the desired rotation from the current orientation of the end effector.
            update_targets (bool): whether to update ik target pos / ori attributes or not

        Returns:
            list: A list of size @num_joints corresponding to the target joint angles.
        """

        des_pos = self.interpolator_pos.get_interpolated_goal()
        ori_error = self.interpolator_ori.get_interpolated_goal()

        targets = (des_pos,  ori_error)

        # convert from target pose in base frame to target pose in bullet world frame
        world_targets = self.bullet_base_pose_to_world_pose(targets)

        # print("[MAIN IK] world_targets: ", world_targets[0], world_targets[1])
        # Converge to IK solution
        arm_joint_pos = None

        for bullet_i in range(self.converge_steps):
            arm_joint_pos = self.inverse_kinematics(world_targets[0], world_targets[1])

            self.sync_ik_robot(arm_joint_pos, sync_last=True)

        return arm_joint_pos

    def bullet_base_pose_to_world_pose(self, pose_in_base):
        """
        Convert a pose in the base frame to a pose in the world frame.

        Args:
            pose_in_base (2-tuple): a (pos, orn) tuple, where orn is in wxyz format

        Returns:
            2-tuple: a (pos, orn) tuple reflecting robot pose in world coordinates, where orn is in wxyz format
        """
        pose_in_base = pose2mat(pose_in_base)

        base_pos_in_world, base_orn_in_world = p.getBasePositionAndOrientation(
            self.ik_robot, physicsClientId=self.bullet_server_id
        )
        base_pos_in_world= np.array(base_pos_in_world)
        base_orn_in_world_wxyz = np.array([base_orn_in_world[3], 
                                           base_orn_in_world[0], 
                                           base_orn_in_world[1], 
                                           base_orn_in_world[2]])

        base_pose_in_world = pose2mat((base_pos_in_world, base_orn_in_world_wxyz))

        pose_in_world = pose_in_A_to_pose_in_B(pose_A=pose_in_base, pose_A_in_B=base_pose_in_world)

        return mat2pose(pose_in_world)
    
    def set_start(self):
        """
        Sets the start state of the controller
        """
        self.update(force=True)
        
        self.reset_goal()
    
    def set_goal(self, action, set_ik=None):
        """
        Sets internal goal state of the controller based on input @action.

        Note that this controller wraps a JointVelocityController, and so determines the desired velocities
        to achieve the inputted pose, and sets its internal setpoint in terms of joint velocities

        Args:
            action (Iterable): Desired relative position / orientation goal state
            set_ik (Iterable): If set, overrides @action and sets the desired absolute joint position goal state
        """
        # Update robot arm state
        self.update()

        # If using deltas:
        if self._use_delta:
            # Remove clipping
            dpos = action[:3]
            dquat = axisangle2quat(action[3:])

            set_pos = dpos * self.user_sensitivity + self.ee_pos

            # TODO: this calc is wrong
            set_ori = set_goal_orientation(
                action[3:], self.ee_ori_mat, orientation_limit=None, set_ori=quat2mat(axisangle2quat(action[3:6]))
            )
            
        # Interpret actions as absolute IK goals
        else:
            set_pos = action[:3]
            set_ori = quat2mat(axisangle2quat(action[3:]))
        
        self.goal_pos = set_pos
        self.goal_ori = set_ori
        
        # Set interpolated goals (eef pos and eef ori)
        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

            self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
            
        if self.interpolator_ori is not None:
            self.interpolator_ori.set_goal(
                mat2quat(self.goal_ori)
            )
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0
            
    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint

        Returns:
             np.array: Command torques
        """
        # Update state
        self.update()

        # Get the updated desired target eef pos
        desired_pos = self.interpolator_pos.get_interpolated_goal()

        # Relative orientation based on difference between current ori and ref
        self.relative_ori = orientation_error(self.ee_ori_mat, self.ori_ref)
        des_ori = self.interpolator_ori.get_interpolated_goal()     # quat
        desired_ori_mat = quat2mat(des_ori)        

        # print("[IK Run controller] desired_pos:", desired_pos)
        # print("[IK Run controller] des ori:", des_ori)

        pos_error = desired_pos - self.ee_pos
        ori_error = orientation_error(desired_ori_mat, self.ee_ori_mat) # error in axis angle

        twist = np.zeros(6)

        twist[:3] = pos_error 
        twist[3:] = ori_error

        # get the jacobian
        dq = self.J_full.T @ np.linalg.solve(self.J_full @ self.J_full.T + self._kp, twist)
        q = self._sim.qpos.copy()
        # append 0s to dq to match q shape
        dq = np.append(dq, np.zeros(q.shape[0] - dq.shape[0]))
        mujoco.mj_integratePos(self._sim._model, q, dq, 1.0)    # 1.0 = integration dt

        # always run superclass call for any cleanups at the end
        super().run_controller()

        # Run controller with given action
        return q[:6]


    def update_initial_joints(self, initial_joints):
        # First, update from the superclass method
        super().update_initial_joints(initial_joints)

        # Then, update the rest pose from the initial joints
        self.rest_poses = list(self.initial_joint_pos)

    def reset_goal(self):
        """
        Resets the goal to the current pose of the robot
        """
        # Also reset interpolators if required
        self.goal_ori = np.array(self.ee_ori_mat)
        self.goal_pos = np.array(self.ee_pos)

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(
                mat2quat(self.goal_ori)
            )
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

        # Sync pybullet state as well
        self.sync_state()

    def _clip_ik_input(self, dpos, rotation):
        """
        Helper function that clips desired ik input deltas into a valid range.

        Args:
            dpos (np.array): a 3 dimensional array corresponding to the desired
                change in x, y, and z end effector position.
            rotation (np.array): relative rotation in scaled axis angle form (ax, ay, az)
                corresponding to the (relative) desired orientation of the end effector.

        Returns:
            2-tuple:

                - (np.array) clipped dpos
                - (np.array) clipped rotation
        """
        # scale input range to desired magnitude
        if dpos.any():
            dpos, _ = clip_translation(dpos, self.ik_pos_limit)

        # Map input to quaternion
        rotation = axisangle2quat(rotation)

        # Clip orientation to desired magnitude
        rotation, _ = clip_rotation_wxyz(rotation, self.ik_ori_limit)

        return dpos, rotation

    @staticmethod
    def _get_current_error(current, set_point):
        """
        Returns an array of differences between the desired joint positions and current
        joint positions. Useful for PID control.

        Args:
            current (np.array): the current joint positions
            set_point (np.array): the joint positions that are desired as a numpy array

        Returns:
            np.array: the current error in the joint positions
        """
        error = current - set_point
        return error

    @property
    def control_limits(self):
        """
        The limits over this controller's action space, as specified by self.ik_pos_limit and self.ik_ori_limit
        and overriding the superclass method

        Returns:
            2-tuple:

                - (np.array) minimum control values
                - (np.array) maximum control values
        """
        max_limit = np.concatenate([self.ik_pos_limit * np.ones(3), self.ik_ori_limit * np.ones(3)])
        return -max_limit, max_limit

    @property
    def pybullet_joint_states(self):
        """
        Returns the current joint states as reported by the pybullet simulation

        Returns:
            list: list of joint states
        """
        return np.array([p.getJointState(self.ik_robot, i)[0] for i in self.bullet_joint_indexes])
    
    @property
    def name(self):
        return "IK_POSE"