# from preference_planner import PreferencePlanner
import ast

# planner = PreferencePlanner()

items = ['chicken', 'rice', 'egg']
portions = [2, 3, 1]
efficiencies = [1, 1, 1]
preference = "I want to eat alternating bites of chicken and rice most of the time but I would like to eat some egg occasionally."
bite_preference = "I want to eat very little food in this bite."
dips = 0.0
history = ['chicken']
bite_size = 0.5

# planner.plan_motion_primitives(items, portions, efficiencies, preference, history, mode="ours")
# planner.plan_motion_primitives(items, portions, efficiencies, preference, bite_preference, history, bite_size, mode="motion_primitive")

response = "Food Items Left: There are 2 portions of chicken, 3 portions of rice, and 1 portion of egg left on the plate.\nStrategy: Since you prefer alternating bites of chicken and rice with occasional egg, I will follow an alternating pattern of chicken and rice primarily, introducing egg after a few alternating bites to ensure variety and adherence to your preference.\nNext bite: Since you just had chicken, the next bite planned will be rice to maintain the alternating pattern.\nNext bite (accounting for efficiency): Given all items have the same efficiency rating of 1, the decision remains to feed rice next, adhering to the alternating preference without needing adjustment for efficiency.\nNext bite as list: ['rice']\nNext bite size: Since the preference is for very little food in this bite, and the current bite size is 0.5, I will reduce the bite size significantly.\nNext bite size as float: 0.3\nNext bite size: TESTING\nNext distance to mouth as float: 7.5\nNext entry angle: TESTING\nNext entry angle as float: 90.0\nNext exit angle: TESTING\nNext exit angle as float: 90.0"

intermediate_response = response.split('Next bite as list:')[1].strip()
feeding_parameters = intermediate_response.split('\n')
next_bites = ast.literal_eval(feeding_parameters[0])
bite_size = ast.literal_eval(feeding_parameters[2].split('Next bite size as float:')[1].strip())
distance_to_mouth = ast.literal_eval(feeding_parameters[4].split('Next distance to mouth as float:')[1].strip())
entry_angle = ast.literal_eval(feeding_parameters[6].split('Next entry angle as float:')[1].strip())
exit_angle = ast.literal_eval(feeding_parameters[8].split('Next exit angle as float:')[1].strip())
print(f"next_bites: {next_bites} (type: {type(next_bites)})")
print(f"bite_size: {bite_size} (type: {type(bite_size)})")
print(f"distance_to_mouth: {distance_to_mouth} (type: {type(distance_to_mouth)})")
print(f"entry_angle: {entry_angle} (type: {type(entry_angle)})")
print(f"exit_angle: {exit_angle} (type: {type(exit_angle)})")