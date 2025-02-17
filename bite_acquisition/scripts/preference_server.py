import ast

import rospy
import rospkg

from std_msgs.msg import String
from feeding_msgs.msg import FoodItem
from feeding_msgs.srv import GetFeedingParam, GetFeedingParamResponse

from preference_planner import PreferencePlanner

class PreferenceServer():
    def __init__(self):
        
        # self.user_preference = "Please feed me all the green beans first. Then I want you to alternate between the foods in the order of rice, then fish, then egg. Give me smaller bites for rice and feed me slower for the fish to give me more time to chew."
        self.user_preference = None
        self.available_food_items = ["mashed potatoes", "corn", "minced meat"]
        self.available_food_items_portions = [3, 3, 3]
        self.history = []
        self.preference_change = True
        self.mode = 'decomposer'
        self.output_directory = rospkg.RosPack().get_path('bite_acquisition') + '/scripts/feeding_bot_output/real_arm_testing'

        self.preference_planner = PreferencePlanner()

        # ROS subscribers
        self.sub_user_preference = rospy.Subscriber("user_preference", String, self.user_preference_cb)

        # Service server
        self.srv_feeding_parameters = rospy.Service('get_feeding_parameters', GetFeedingParam, self.get_feeding_parameters_cb)
        rospy.loginfo(f"Service server started")
        
    def update_feeding_parameters(self):

        # TODO: Need to do something about this such that it won't rerun when I call it again

        self.feeding_sequence = self.preference_planner.plan(
            self.available_food_items, 
            self.available_food_items_portions, 
            self.user_preference, 
            self.history,
            self.preference_change,
            0,
            self.mode,
            self.output_directory
            )
        
        self.preference_change = False
        
    def user_preference_cb(self, msg):
        """
        Callback function for user preference
        """
        self.user_preference = msg.data
        rospy.loginfo(f"Obtained user preference")

        self.preference_change = True

        # Get updated feeding parameters
        rospy.loginfo("Calling planner")
        self.update_feeding_parameters()


    def get_feeding_parameters_cb(self, request):
        """
        Callback function for getting feeding parameters
        """
        rospy.loginfo("Getting feeding parameters")

        response = GetFeedingParamResponse()

        if self.user_preference is None:
            rospy.logwarn("User preference not set. Please set user preference first.")
            response.success = False
            return response
        else:
            current_history = request.current_history # This should give me the current history after the robot fed. and the other updated variables
            current_history = ast.literal_eval(current_history)

            self.available_food_items_portions = request.food_item_portions
            self.history = current_history

            self.update_feeding_parameters()

            response.feeding_sequence = [FoodItem(food_item, bite_size, distance_to_mouth, exit_angle, transfer_speed) for food_item, bite_size, distance_to_mouth, exit_angle, transfer_speed in self.feeding_sequence]
            response.success = True

        return response
    
if __name__ == "__main__":

    rospy.init_node("feeding_param_server")
    rospy.loginfo(f"Starting {rospy.get_name()} node...")

    preference_server = PreferenceServer()
    # # to avoid talking to moonshine
    # preference_server.user_preference = "Give me the chicken and rice"
    # preference_server.preference_change = False

    while not rospy.is_shutdown():
        rospy.spin()
    
    rospy.loginfo(f"Exiting node {rospy.get_name()}...")