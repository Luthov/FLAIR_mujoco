from preference_planner import PreferencePlanner
from preferences import luke_preferences, interview_food_items
preference_planner = PreferencePlanner()
current_motion_params = [
    [('rice', 3.0, 7.5, 90.0, 5.0), ('fish', 5.0, 7.5, 90.0, 3.0), ('egg', 5.0, 7.5, 90.0, 5.0), ('green beans', 5.0, 7.5, 90.0, 5.0)],
    [('rice', 5.0, 7.5, 95.0, 4.0), ('chicken', 5.0, 7.5, 90.0, 5.0), ('egg', 5.0, 7.5, 90.0, 5.0), ('green beans', 5.0, 7.5, 90.0, 5.0)],
    [('rice', 7.0, 7.5, 90.0, 5.0), ('chicken', 3.0, 7.5, 90.0, 5.0), ('egg', 5.0, 8.5, 90.0, 5.0), ('cucumber', 5.0, 7.5, 90.0, 5.0)],
    [('rice', 7.0, 7.5, 90.0, 5.0), ('chicken', 3.0, 7.5, 85.0, 5.0), ('cucumber', 5.0, 7.5, 90.0, 5.0)],
    [('mashed potatoes', 7.0, 7.5, 90.0, 5.0), ('steak', 7.0, 8.0, 90.0, 5.0), ('green beans', 5.0, 7.5, 90.0, 7.0), ('carrots', 5.0, 7.5, 90.0, 7.0)],
    [('rice', 3.0, 7.5, 90.0, 3.0), ('beef', 3.0, 7.5, 90.0, 3.0), ('broccoli', 3.0, 7.5, 90.0, 3.0)],
    [('mashed potatoes', 7.0, 7.5, 90.0, 5.0), ('meatballs', 5.0, 7.5, 100.0, 5.0), ('green beans', 5.0, 7.5, 90.0, 5.0)],
    [('mashed potatoes', 7.0, 7.5, 90.0, 5.0), ('salmon', 7.0, 8.5, 90.0, 5.0), ('broccoli', 7.0, 7.5, 90.0, 5.0)],
    [('rice', 7.0, 7.5, 95.0, 3.0), ('pork cutlet', 6.0, 7.5, 90.0, 5.0), ('cabbage', 5.0, 7.5, 90.0, 5.0)],
    [('rice', 3.0, 7.5, 80.0, 5.0), ('chicken', 3.0, 7.5, 80.0, 5.0), ('mixed vegetables', 3.0, 7.5, 80.0, 5.0)]
]
food_items = ["rice", "chicken", "cucumber"]
food_portion_rounded = [3, 3, 3]
bite_history = []
update_motion_params = True
mode = 'decomposer'
output_directory = 'feeding_bot_output/PLEASE/interrupt_testing/'

for preference_idx in [4]: # range(2,len(luke_preferences)):
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