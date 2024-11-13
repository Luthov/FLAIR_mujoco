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