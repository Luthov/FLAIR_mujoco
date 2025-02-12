from preference_planner import PreferencePlanner

preference_planner = PreferencePlanner()

food_items = ["rice", "chicken", "cucumber"]
food_portion_rounded = [3, 3, 3]
user_preference = "Feed me alternating bites of rice and chicken. I don't want any vege. I want bigger bites of rice and smaller bites for chicken. Also tilt the spoon lower for chicken."
bite_history = []
update_motion_params = True
preference_idx = 0
mode = 'decomposer'
output_directory = 'feeding_bot_output/PLEASE/'
next_bite, bite_size, distance_to_mouth, exit_angle, transfer_speed, token_data = preference_planner.plan(
                        food_items, 
                        food_portion_rounded, 
                        user_preference, 
                        bite_history,
                        update_motion_params,
                        preference_idx,
                        mode,
                        output_directory
                        )