from preference_planner import PreferencePlanner

planner = PreferencePlanner()

items = ['chicken', 'rice', 'egg']
portions = [2, 3, 1]
efficiencies = [1, 1, 1]
preference = "I want to eat alternating bites of chicken and rice most of the time but I would like to eat some egg occasionally."
bite_preference = "I want to eat very little food in this bite."
dips = 0.0
history = ['chicken']
bite_size = 0.5

# planner.plan_motion_primitives(items, portions, efficiencies, preference, history, mode="ours")
planner.plan_motion_primitives(items, portions, efficiencies, preference, bite_preference, history, bite_size, mode="motion_primitive")

# response = "Food Items Left: There are 2 portions of chicken, 3 portions of rice, and 1 portion of egg left on the plate.\nStrategy: Since you prefer alternating bites of chicken and rice with occasional egg, I will follow an alternating pattern of chicken and rice primarily, introducing egg after a few alternating bites to ensure variety and adherence to your preference.\nNext bite: Since you just had chicken, the next bite planned will be rice to maintain the alternating pattern.\nNext bite (accounting for efficiency): Given all items have the same efficiency rating of 1, the decision remains to feed rice next, adhering to the alternating preference without needing adjustment for efficiency.\nNext bite as list: ['rice']\nNext bite size: Since the preference is for very little food in this bite, and the current bite size is 0.5, I will reduce the bite size significantly.\nNext bite size as float: 0.3"
# intermediate_response = response.split('Next bite as list:')[1] #.strip()
# next_bites = intermediate_response.split('Next bite size:')[0].strip()
# bite_size = intermediate_response.split('Next bite size as float:')[1].strip()
# print(f"next_bites: {next_bites} \n bite_size: {bite_size}")