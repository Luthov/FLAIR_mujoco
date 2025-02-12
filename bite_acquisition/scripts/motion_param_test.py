from preference_planner import PreferencePlanner
from preferences import luke_preferences, interview_food_items
preference_planner = PreferencePlanner()

bite_history = []
update_motion_params = True
mode = 'decomposer'
output_directory = 'feeding_bot_output/luke/10_bite_testing/'

for preference_idx in range(len(luke_preferences)):
    if len(interview_food_items[preference_idx]) == 4:
        food_portion_rounded = [3, 3, 3, 3]
    else:
        food_portion_rounded = [3, 3, 3]
    preference_planner.plan(
                            interview_food_items[preference_idx], 
                            food_portion_rounded, 
                            luke_preferences[preference_idx], 
                            bite_history,
                            # current_motion_params[preference_idx],
                            update_motion_params,
                            preference_idx,
                            mode,
                            output_directory
                            )