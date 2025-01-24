import ast

import rospy

from std_msgs.msg import String
from feeding_msgs.srv import GetFeedingParam, GetFeedingParamResponse

from preference_planner import PreferencePlanner

class PreferenceServer():
    def __init__(self):
        
        self.user_preference = None
        self.available_food_items = ['chicken', 'rice', 'broccoli']
        self.available_food_items_portions = [2, 2, 2]
        self.efficiency_scores = [1, 1, 1]
        self.history = []
        self.mode = 'no_decomposer'
        self.output_directory = '/home/luthov/school/fyp/feeding_ws/src/feeding/task_planner/FLAIR_mujoco/bite_acquisition/scripts/feeding_bot_output/real_arm_testing/'

        self.preference_planner = PreferencePlanner()

        # ROS subscribers
        self.sub_user_preference = rospy.Subscriber("user_preference", String, self.user_preference_cb)

        # Service server
        self.srv_feeding_parameters = rospy.Service('get_feeding_parameters', GetFeedingParam, self.get_feeding_parameters_cb)
        rospy.loginfo(f"Service server started")
        
    def update_feeding_parameters(self):

        self.next_bite, self.bite_size, self.distance_to_mouth, self.exit_angle, self.transfer_speed, _ = self.preference_planner.plan(
            self.available_food_items, 
            self.available_food_items_portions, 
            self.efficiency_scores, 
            self.user_preference, 
            self.history,
            0,
            self.mode,
            self.output_directory
            )
        
    def user_preference_cb(self, msg):
        """
        Callback function for user preference
        """
        self.user_preference = msg.data
        rospy.loginfo(f"Obtained user preference")

        # Get updated feeding parameters
        rospy.loginfo("Calling planner")
        self.update_feeding_parameters()


    def get_feeding_parameters_cb(self, request):
        """
        Callback function for getting feeding parameters
        """
        rospy.loginfo("Getting feeding parameters")

        if self.user_preference is None:
            rospy.logwarn("User preference not set. Please set user preference first.")
            return None
        else:
            current_history = request.current_history # This should give me the current history after the robot fed. and the other updated variables
            current_history = ast.literal_eval(current_history)
            self.available_food_items_portions = request.food_item_portions
            # if current_history != self.history:
            #     rospy.logerr(f"Current history: {current_history}")
            #     rospy.logerr(f"self.history: {self.history}")
            #     rospy.logwarn("History has changed. Updating feeding parameters...")
            self.history = current_history
            self.update_feeding_parameters()

            response = GetFeedingParamResponse()
            response.next_bite = self.next_bite
            response.bite_size = self.bite_size
            response.distance_to_mouth = self.distance_to_mouth
            response.exit_angle = self.exit_angle
            response.transfer_speed = self.transfer_speed
            response.user_preference = self.user_preference

        return response
    
if __name__ == "__main__":

    rospy.init_node("feeding_param_server")
    rospy.loginfo(f"Starting {rospy.get_name()} node...")

    preference_server = PreferenceServer()

    while not rospy.is_shutdown():
        rospy.spin()
    
    rospy.loginfo(f"Exiting node {rospy.get_name()}...")