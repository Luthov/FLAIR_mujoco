import rospy
import numpy as np
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import PoseStamped

from robot_controller.mujoco_action_controller import MujocoRobotController

if __name__ == "__main__":
    rospy.init_node('mujoco_controller')
    robot_controller = MujocoRobotController()
    pose = np.array([[0, 0, 0, 0.05],
                    [0, 0, 0, 0.05],
                    [0, 0, 0, 0.05],
                    [0, 0, 0, 1]])

    goal_point = PoseStamped()
    goal_point.pose.position.x = 0.3507
    goal_point.pose.position.y = 0.0512
    goal_point.pose.position.z = 0.0373 + 0.2

    goal_point.pose.orientation.x = 0.9238795325050545
    goal_point.pose.orientation.y = 0.0
    goal_point.pose.orientation.z = 0.3826834323625085
    goal_point.pose.orientation.w = 0.0

    acq_pose = PoseStamped()
    acq_pose.pose.position.x = 0.25
    acq_pose.pose.position.y = 0.0
    acq_pose.pose.position.z = 0.3

    acq_pose.pose.orientation.x = 0.9238795325050545
    acq_pose.pose.orientation.y = 0.0
    acq_pose.pose.orientation.z = 0.3826834323625085
    acq_pose.pose.orientation.w = 0.0

    # Convert from euler angles to quaternions
    r = Rotation.from_euler('xyz', [-180, -90, 0], degrees=True)
    quat = r.as_quat()

    transfer_pose = PoseStamped()
    transfer_pose.pose.position.x = 0.4
    transfer_pose.pose.position.y = 0.0
    transfer_pose.pose.position.z = 0.4

    transfer_pose.pose.orientation.x = quat[0]
    transfer_pose.pose.orientation.y = quat[1]
    transfer_pose.pose.orientation.z = quat[2]
    transfer_pose.pose.orientation.w = quat[3]

    print(f"trasnfer_pose: {transfer_pose}")

    # copy = goal_point

    # print(f"goal_point: {goal_point} | copy: {copy}")

    # robot_controller.rotate_eef(0.5)
    
    input("Press Enter to move to pose")
    robot_controller.move_to_pose(goal_point)

    # goal_point = PoseStamped()
    # goal_point.pose.position.x = 0.4507
    # goal_point.pose.position.y = -0.0512
    # goal_point.pose.position.z = 0.0373 + 0.2

    # goal_point.pose.orientation.x = 0.9238795325050545
    # goal_point.pose.orientation.y = 0.0
    # goal_point.pose.orientation.z = 0.3826834323625085
    # goal_point.pose.orientation.w = 0.0

    # # copy = goal_point

    # # print(f"goal_point: {goal_point} | copy: {copy}")
    
    # input("Press Enter to move to end pose")
    # robot_controller.move_to_pose(goal_point)

    input("Press Enter to move to Bite Acq Pose")
    robot_controller.move_to_acq_pose(acq_pose)

    input("Press Enter to move to Bite Transfer Pose")
    robot_controller.move_to_transfer_pose(transfer_pose)