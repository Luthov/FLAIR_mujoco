from feeding_msgs.msg import BiteTransferAction, BiteTransferGoal 
import actionlib
import rospy

class ThreeDDFAClient:
    def __init__(self):
        self.client = actionlib.SimpleActionClient('start_signal', BiteTransferAction)
        self.client.wait_for_server()
    
    def start_mouth_tracking(self):
        goal = BiteTransferGoal()
        print(goal)
        goal.distance_to_mouth = 10
        goal.exit_angle = 90
        goal.transfer_speed = 6
        self.client.send_goal(goal)
        self.client.wait_for_result()
        return self.client.get_result()

if __name__ == '__main__':
    rospy.init_node("3ddfa_client_node")
    client = ThreeDDFAClient()
    client.start_mouth_tracking()