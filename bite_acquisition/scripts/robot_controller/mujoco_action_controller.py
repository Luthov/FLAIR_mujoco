import rospy
import actionlib

from bite_acquisition.msg import mujoco_action_serverAction, mujoco_action_serverGoal

from .base import RobotController

class MujocoRobotController(RobotController):
    def __init__(self):
        self.client = actionlib.SimpleActionClient('mujoco_action_server', mujoco_action_serverAction)
        self.client.wait_for_server()

    def reset(self):
        self.move_to_acq_pose()

    def move_to_pose(self, pose):
        goal = mujoco_action_serverGoal()
        goal.function_name = "move_to_pose"
        goal.goal_point.header.frame_id = "world"
        goal.goal_point.pose.position.x = pose.pose.position.x
        goal.goal_point.pose.position.y = pose.pose.position.y
        goal.goal_point.pose.position.z = pose.pose.position.z
        goal.goal_point.pose.orientation.x = pose.pose.orientation.x
        goal.goal_point.pose.orientation.y = pose.pose.orientation.y
        goal.goal_point.pose.orientation.z = pose.pose.orientation.z
        goal.goal_point.pose.orientation.w = pose.pose.orientation.w

        self.client.send_goal(goal)
        self.client.wait_for_result()
        return self.client.get_result()

    def move_to_acq_pose(self):
        goal = mujoco_action_serverGoal()
        goal.function_name = "move_to_acq_pose"
        goal.goal_point.header.frame_id = "world"
        goal.goal_point.pose.position.x = 0.0
        goal.goal_point.pose.position.y = 0.0
        goal.goal_point.pose.position.z = 0.0
        goal.goal_point.pose.orientation.x = 0.0
        goal.goal_point.pose.orientation.y = 0.0
        goal.goal_point.pose.orientation.z = 0.0
        goal.goal_point.pose.orientation.w = 1.0

        self.client.send_goal(goal)
        self.client.wait_for_result()
        return self.client.get_result()

    def move_to_transfer_pose(self):
        goal = mujoco_action_serverGoal()
        goal.function_name = "move_to_transfer_pose"
        goal.goal_point.header.frame_id = "world"
        goal.goal_point.pose.position.x = 0.0
        goal.goal_point.pose.position.y = 0.0
        goal.goal_point.pose.position.z = 0.0
        goal.goal_point.pose.orientation.x = 0.0
        goal.goal_point.pose.orientation.y = 0.0
        goal.goal_point.pose.orientation.z = 0.0
        goal.goal_point.pose.orientation.w = 1.0

        self.client.send_goal(goal)
        self.client.wait_for_result()
        return self.client.get_result()

if __name__ == '__main__':
    rospy.init_node('mujoco_action_client')
    robot_controller = MujocoRobotController()