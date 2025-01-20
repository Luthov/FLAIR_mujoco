from three_ddfa_ros.msg import StartActionAction, StartActionGoal, StartActionFeedback, StartActionResult
import actionlib
import rospy

class ThreeDDFAClient:
    def __init__(self):
        self.client = actionlib.SimpleActionClient('start_signal', StartActionAction)
        self.client.wait_for_server()
    
    def start_mouth_tracking(self):
        goal = StartActionGoal()
        print(goal)
        goal.start = True
        self.client.send_goal(goal)
        self.client.wait_for_result()
        return self.client.get_result()

if __name__ == '__main__':
    rospy.init_node("3ddfa_client_node")
    client = ThreeDDFAClient()
    client.start_mouth_tracking()