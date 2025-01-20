# THIS IS SCUFFED
import sys
sys.path.insert(0, '/home/luthov/school/fyp/feeding_ws/src/feeding/task_planner/FLAIR_mujoco/bite_acquisition/scripts')

from robot_controller.xarm6_controller import XArm6RobotController

if __name__ == "__main__":
    robot_controller = XArm6RobotController()

    input("Press ENTER to test bite transfer")
    robot_controller.execute_bite_transfer()
