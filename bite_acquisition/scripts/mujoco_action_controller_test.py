import rospy

from robot_controller.mujoco_action_controller import MujocoRobotController

if __name__ == "__main__":
    rospy.init_node('mujoco_controller')
    robot_controller = MujocoRobotController()

    input('Press enter to move to acquisition position...')
    robot_controller.move_to_acq_pose()

    input('Press enter to move to transfer position...')
    robot_controller.move_to_transfer_pose()

    input('Press enter to reset the robot...')
    robot_controller.reset()