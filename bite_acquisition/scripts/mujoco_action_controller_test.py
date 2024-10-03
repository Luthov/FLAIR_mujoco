import rospy

from robot_controller.mujoco_action_controller import MujocoRobotController

if __name__ == "__main__":
    rospy.init_node('mujoco_controller')
    robot_controller = MujocoRobotController()

    input("Press Enter to move to test pose")
    robot_controller.move_to_pose()