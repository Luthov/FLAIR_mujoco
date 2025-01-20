import ast

import rospy

from std_msgs.msg import String
from feeding_msgs.srv import GetFeedingParam, GetFeedingParamRequest

if __name__ == "__main__":
    rospy.init_node("feeding_param_client")
    rospy.loginfo(f"Starting {rospy.get_name()} node...")

    rospy.wait_for_service("get_feeding_parameters")

    available_food_items = ['chicken', 'rice', 'broccoli']
    food_item_portions = [2, 2, 2]
    bite_history = []

    # Get scooping point
    try:
        for i in range(9):
            srv_get_scooping_point = rospy.ServiceProxy("get_feeding_parameters", GetFeedingParam)

            rospy.loginfo("Getting feeding parameters...")
            req_feeding_params = GetFeedingParamRequest()
            req_feeding_params.current_history = str(bite_history)
            food_item_portions_rounded = [round(x) for x in food_item_portions]
            req_feeding_params.food_item_portions = food_item_portions_rounded
            resp_feeding_params = srv_get_scooping_point(req_feeding_params)

            next_bite = resp_feeding_params.next_bite
            bite_size = resp_feeding_params.bite_size
            distance_to_mouth = resp_feeding_params.distance_to_mouth
            exit_angle = resp_feeding_params.exit_angle
            transfer_speed = resp_feeding_params.transfer_speed
            # rospy.loginfo(f"Next bite: {next_bite}")
            # rospy.loginfo(f"Bite size: {bite_size}")
            # rospy.loginfo(f"Distance to mouth: {distance_to_mouth}")
            # rospy.loginfo(f"Exit angle: {exit_angle}")
            # rospy.loginfo(f"Transfer speed: {transfer_speed}")
            rospy.logwarn(f"HISTORY: {bite_history}")

            for idx in range(len(available_food_items)):
                if (next_bite == available_food_items[idx]):
                    food_item_portions[idx] -= 0.6
                    break

            bite_history.append([next_bite, bite_size, distance_to_mouth, exit_angle, transfer_speed])
    
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s"%e)
    
    rospy.loginfo(f"Completed service call! Exiting node {rospy.get_name()}...")
