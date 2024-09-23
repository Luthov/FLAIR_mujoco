import rospy
from robot_controller.mujoco_controller import MujocoRobotController

robot_controller = MujocoRobotController()

if __name__ == "__main__":
    rospy.init_node("moveit_planner")

    # while not rospy.is_shutdown():
    # robot_controller.move_to_acq_pose()
    robot_controller.move_to_transfer_pose()
    print("DONE")
    
    while True:
        robot_controller.env._sim.step()