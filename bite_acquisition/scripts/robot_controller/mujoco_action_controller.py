import rospy
import actionlib

from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from bite_acquisition.msg import MujocoActionServerAction, MujocoActionServerGoal

from .base import RobotController

# TODO: Luke: Class should have same name as script name ideally
class MujocoRobotController(RobotController):
    def __init__(self):
        # TODO: Luke: Rename action file to MujocoActionServer (not mujoco_action_server)
        # TODO: Luke: Ideally place any relevant actions/srvs/msgs under feeding_msgs for consistency
        self.client = actionlib.SimpleActionClient('mujoco_action_server', MujocoActionServerAction)
        self.client.wait_for_server()

     
    def reset(self):
    # TODO: Luke: Can define a reset pose which is fixed in the action server
    # name the pose appropriately
        self.move_to_reset_pos()

    def move_to_pose(self, pose):
        goal = MujocoActionServerGoal()
        goal.function_name = "move_to_pose"
        goal.goal_point.header.frame_id = "world"
        goal.goal_point = pose

        self.client.send_goal(goal)
        self.client.wait_for_result()
        return self.client.get_result()

    def move_to_acq_pose(self, pose):
        """
        Calls the action server to move the robot to the acquisition pose.
        Note that the acquisition pose is defined in the action server.
        TODO: Luke: acquisition pose will change based on the bowl pose 
                    should allow this function to take in a pose
        """
        goal = MujocoActionServerGoal()
        goal.function_name = "move_to_acq_pose"
        goal.goal_point.header.frame_id = "world"
        goal.goal_point = pose

        self.client.send_goal(goal)
        self.client.wait_for_result()
        return self.client.get_result()

    def move_to_reset_pos(self):
        """
        Calls the action server to move the robot to the reset pose.
        Note that the reset pose is defined in the action server.
        """
        goal = MujocoActionServerGoal()
        goal.function_name = "move_to_reset_pos"
        goal.goal_point.header.frame_id = "world"

        self.client.send_goal(goal)
        self.client.wait_for_result()
        return self.client.get_result()

    def move_to_transfer_pose(self, pose):
        """
        Calls the action server to move the robot to the transfer pose.
        Note that the acquisition pose is defined in the action server.
        TODO: Luke: transfer pose will change based on the user's mouth pose, 
                    should allow this function to take in a pose
        """
        goal = MujocoActionServerGoal()
        goal.function_name = "move_to_transfer_pose"
        goal.goal_point.header.frame_id = "world"
        goal.goal_point = pose

        self.client.send_goal(goal)
        self.client.wait_for_result()
        return self.client.get_result()
    
    def execute_scooping(self, scooping_trajectory, food_pose):
        """
        Calls the action server to execute the scooping trajectory.
        """
        goal = MujocoActionServerGoal()
        goal.function_name = "execute_scooping"
        
        msg_scooping_trajectory = Float32MultiArray()
        flattened_traj = [item for wp in scooping_trajectory for item in wp]
        msg_scooping_trajectory.data = flattened_traj

        # Set layout information
        msg_scooping_trajectory.layout.dim.append(MultiArrayDimension())
        msg_scooping_trajectory.layout.dim[0].label = "waypoints"
        msg_scooping_trajectory.layout.dim[0].size = len(scooping_trajectory)
        msg_scooping_trajectory.layout.dim[0].stride = 7    # 7 elements per waypoint

        msg_scooping_trajectory.layout.dim.append(MultiArrayDimension())
        msg_scooping_trajectory.layout.dim[1].label = "elements_per_waypoint"
        msg_scooping_trajectory.layout.dim[1].size = 7
        msg_scooping_trajectory.layout.dim[1].stride = 1

        goal.scooping_trajectory = msg_scooping_trajectory
        goal.goal_point.header.frame_id = "world"
        goal.goal_point.pose.position.x = food_pose[0]
        goal.goal_point.pose.position.y = food_pose[1]
        goal.goal_point.pose.position.z = food_pose[2]

        self.client.send_goal(goal)
        self.client.wait_for_result()
        return self.client.get_result()
    
    def rotate_eef(self, angle):
        goal = MujocoActionServerGoal()
        goal.function_name = "rotate_eef"
        goal.angle = angle

        self.client.send_goal(goal)
        self.client.wait_for_result()
        return self.client.get_result()

if __name__ == '__main__':
    rospy.init_node('mujoco_action_client')
    robot_controller = MujocoRobotController()