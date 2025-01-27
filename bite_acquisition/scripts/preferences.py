## User preferences with different food items
user_preference = "I want to eat all the pasta first, followed by a bite of steak, and end each cycle with a mix of greens and cherry tomatoes."
user_preference = "I prefer to eat alternating bites of sushi and avocado slices, but I want an occasional crunchy tempura piece to surprise me."
user_preference = "Serve me the soup first, but make sure I get a spoonful of croutons with every other bite. I’ll finish with the roasted vegetables."
user_preference = "I want to alternate between spicy curry and naan, with a small piece of paneer cheese added to every third bite."
user_preference = "Give me one massive bite of burger followed by three tiny fries, and then repeat. End with a small pickle slice occasionally."

## Bite preferences with different food items
bite_preference = "I want massive bites of steak but smaller bites of greens and cherry tomatoes mixed in."
bite_preference = "I’d like my sushi served whole, but the avocado and tempura can be in smaller, delicate portions."
bite_preference = "I prefer large, hearty spoonfuls of soup with medium-sized croutons and very tiny pieces of roasted vegetables on the side."
bite_preference = "I like naan in large chunks, curry in medium spoonfuls, and paneer in small cubes."
bite_preference = "I want large burger bites, tiny fries, and an occasional small pickle piece for contrast."

##################################################################################
# Lists to store user preferences and bite preferences with the same food items
user_preferences_same_food = [
    "Start with a big bite of rice mixed with chicken, followed by two small bites of egg, and finish with a pinch of plain rice.",
    "I want to eat alternating bites of egg and rice for a while, then switch to just chicken until the plate is almost empty.",
    "Serve me rice and egg together in the first few bites, then chicken in a standalone bite to reset my palate.",
    "I want small bites of rice with egg until there’s only half the rice left, then finish the plate with chicken-only bites.",
    "I like bites to alternate between chicken, rice, and egg in a clockwise rotation, but add an extra piece of chicken every fourth bite."
]

bite_preferences_same_food = [
    "I want medium-sized bites of chicken, very tiny bites of egg, and rice served in generous spoonfuls.",
    "I’d like my chicken in large, hearty chunks, egg in delicate slivers, and rice in bite-sized scoops.",
    "Serve rice in tiny portions, egg in medium-sized bites, and chicken in large chunks with crispy edges.",
    "I prefer all my rice to be served in one big portion at the start, then small, even bites of chicken and egg for the rest.",
    "I want each bite to be a mix of chicken, egg, and rice, but the chicken should dominate in size and flavor."
]
##################################################################################

# Lists to store user preferences and bite preferences (HOSPITAL FOOD)
user_preferences_hospital = [
    "I want to start with a spoonful of mashed potatoes, then have a small bite of chicken breast, and finish with a taste of steamed carrots.",
    "Alternate between bites of green beans and turkey slices, but make sure I get a spoonful of gravy-covered mashed potatoes every third bite.",
    "Serve me all the soup first, followed by alternating bites of plain rice and baked fish, and end with a nibble of bread roll.",
    "I like to mix bites of rice and steamed broccoli together, with a small slice of grilled chicken every fourth bite.",
    "Give me alternating bites of oatmeal and scrambled eggs for the first half, then finish with a couple of bites of fruit salad."
]

bite_preferences_hospital = [
    "I want big spoonfuls of mashed potatoes, medium-sized bites of chicken, and very tiny bites of carrots.",
    "I’d like my turkey sliced thin and in small pieces, green beans in moderate bites, and mashed potatoes served in generous scoops.",
    "I prefer large chunks of baked fish, small bites of broccoli, and rice in medium spoonfuls.",
    "Serve the soup in big hearty spoonfuls, but keep the bread roll in small bite-sized pieces.",
    "I want small spoonfuls of oatmeal, large chunks of scrambled eggs, and fruit salad in tiny bites for a fresh finish."
]

icorr_preferences = [
    "Feed me all the rice first, then alternate between chicken and vegetables",
    "I want alternate bites of chicken and rice. I prefer to be fed larger bites and for the spoon to be further away from me.",
    "I only want meat. Tilt the spoon slightly when feeding me. Feed me with larger bites", # Rerun idx 2
    "Start with the vegetables, then the meat. Keep the bites small.",
    "Alternate between rice and vegetables, but do not feed me chicken. Keep the spoon far from my mouth.", # Rerun idx 4
    "Feed me all the chicken first, then the rice. Use smaller bites and be careful not to tilt the spoon too much.",
    "I prefer alternate bites of rice, chicken and vegetables. Do not repeat any bites. Feed me evenly without tilting the spoon.",
    "Give me two bites of meat first, then alternate between vegetables and rice. Make sure to tilt the spoon a little higher", # Rerun idx 7
    "I want all the vegetables first, followed by alternating bites of chicken and rice. Keep the bites medium-sized and keep the spoon far from me.",
    "Avoid the vegetables and give me only rice and chicken. Keep the bites small and tilt the spoon slightly.",
    "Start with the chicken, then the vegetables, and end with the rice. Keep the spoon close to me.",
    "Feed me only the rice, one spoonful at a time. Keep the bites large.",
    "Alternate between rice, chicken, and vegetables. Keep the spoon at a distance and use small bites.",
    "I want one bite of chicken followed by two bites of rice. Feed me with tilted spoonfuls.", # Rerun idx 13
    "Feed me vegetables first, then alternate rice and meat. Use small bites and tilt the spoon slightly. Also keep the spoon close to me.",
    "Give me larger bites of chicken, followed by smaller bites of rice. Feed me with a tilt in the spoon and do not come too close to me.", # Rerun idx 15
    "Feed me rice first, then alternate between chicken and vegetables. Keep the spoon tilted slightly upwards.",
    "I only want vegetables. Feed me in small bites..",
    "Start with a bite of vegetables, then alternate between chicken and rice.",
    # "I have no preference in the sequence, but I prefer the spoon to be closer to me."
    # "I want all the rice first, then alternate between chicken and vegetables. Give me larger bites of chicken. Also feed me slower."
    "我先要吃完所有的饭，然后鸡肉和蔬菜交替喂。鸡肉要大口一点，可以吗？还有，喂慢一点。"
    # "I want all the rice first, then chicken and veg alternate, ah. Chicken bigger bite, okay? And feed slower, can?"
    # "I want all the rice first, then chicken and veg alternate, ah. Chicken I want bigger bite and can you also please feed slower for everything"
]
icorr_food_items = [
    ["rice", "chicken", "carrots"],
    ["chicken", "rice", "peas"],
    ["beef", "potatoes", "salad"],
    ["spinach", "beef", "peas"],
    ["rice", "spinach", "broccoli"],
    ["chicken", "rice", "peas"],
    ["rice", "chicken", "carrots"],
    ["beef", "spinach", "rice"],
    ["carrots", "chicken", "rice"],
    ["rice", "chicken", "peas"],
    ["chicken", "spinach", "rice"],
    ["rice", "peas", "corn"],
    ["rice", "chicken", "carrots"],
    ["chicken", "rice", "peas"],
    ["spinach", "rice", "beef"],
    ["chicken", "rice", "peas"],
    ["rice", "chicken", "spinach"],
    ["beef", "rice", "cabbage"],
    ["carrots", "chicken", "rice"],
    ["rice", "chicken", "peas"]
]

# Interview preferences
interview_food_items = [
    ["rice", "fish", "egg", "green beans"],
    ["rice", "chicken", "egg", "green beans"],
    ["rice", "chicken", "egg", "cucumber"],
    ["rice", "chicken", "cucumber"],
    ["mashed potatoes", "steak", "green beans", "carrots"],
    ["rice", "beef", "broccoli"],
    ["mashed potatoes", "meatballs", "green beans"],
    ["mashed potatoes", "salmon", "broccoli"],
    ["rice", "pork cutlet", "cabbage"],
    ["rice", "chicken", "mixed vegetables"]
    ]

hau_wen_interview_preferences = ["alternate between rice and other food.",
                         "alternate between rice and other food. leave some chicken at the end of meal",
                         "alternate between rice and other food.",
                         "alternate between rice and other food. finish cucumber early leave more meat at the end",
                         "alternate between mashed potatoes and other food. leave more steak at the end",
                         "alternate between rice and other foods. leave more beef at the end",
                         "alternate between mashed potatoes and other food.",
                         "alternate between salmon and other food.",
                         "alternate between rice and other food. dont have to finish all the cabbage",
                         "alternate between rice and other food. leave more chicken at the end."]

yi_heng_interview_preferences = [
    "I'd like to have some fish first, then some rice. After that, the egg and green beans. Please repeat that sequence",
    "I'd like to have some chicken first, then some rice. After that, the egg and green beans. Please repeat that sequence",
    "I'd like to have some chicken first, then some rice. After that, the egg and cucumber. Please repeat that sequence",
    "I'd like to have some chicken first, then some rice and finally some cucumber. Please repeat that sequence",
    "I'd like some steak first, then the mashed potatoes, then the beans and carrots. Please repeat that sequence",
    "I'd like to have beef first, followed by the broccoli and finally the rice. Please repeat that sequence",
    "I'd like to have the meatballs first, then the mashed potatoes and finally the green beans. Please repeat the sequence.",
    "I'll have the salmon, then the broccoli and mashed potatoes. Please repeat the sequence",
    "I'll have the pork cutlet, followed by the rice and cabbage. Please repeat the sequence",
    "I'll have the chicken, followed by rice and then, mixed vegetables. Please repeat the sequence"
    ]

yi_heng_modified_interview_preferences = [
    "I'd like to be first fed fish, then rice, then egg and finally green beans in that order",
    "",
    "I'd like to be fed chicken first, followed by rice, egg and cucumber in that order",
    "I'd like to be fed chicken first, then rice and finally cucumber in that order",
    "",
    "I'd like to have beef first, then broccoli, then rice in that order",
    "",
    "I'd like to have some salmon, then broccoli and then mashed potatoes in that order",
    "I'd like to have some pork cutlet, then rice and cabbage in that order",
    "I'd like to have some chicken first, then rice and then vegetables"
]

ben_interview_preferences = [
    "I would like to be fed with rice and the other foods in an alternating. The other foods can be any order but avoid repeating the foods. If there is no more rice, then feed me the vegetables first then the, egg and finally the meat. If there is only rice left, then scoop it until i say stop",
    "I would like to be fed with rice and the other foods in an alternating. The other foods can be any order but avoid repeating the foods. If there is no more rice, then feed me the vegetables first then the, egg and finally the meat. If there is only rice left, then scoop it until i say stop",
    "I would like to be fed with rice and the other foods in an alternating. The other foods can be any order but avoid repeating the foods. If there is no more rice, then feed me the vegetables first then the, egg and finally the meat. If there is only rice left, then scoop it until i say stop",
    "I would like to be fed with rice and the other foods in an alternating. The other foods can be any order but avoid repeating the foods. If there is no more rice, then feed me the vegetables first then the, egg and finally the meat. If there is only rice left, then scoop it until i say stop",
    "I'd like to start with the steak. Then please alternate between all the other food on the plate. Ensure that I finish the green beans and carrots first, followed by the mash potatoes because I'd like my last few bites to be the steak. Serve the bite after the steak slowly as i want to savour the meat. ",
    "I would like to be fed with rice and the other foods in an alternating. The other foods can be any order but avoid repeating the foods. If there is no more rice please ensure that my last bite is the beef. If there is only rice left, then scoop it until i say stop",
    "serve me one meatball first. Then alternate between the mashed potatoes and the green beans unless i ask for a meat ball. After each meatball bite, allow me more time before the next bite. ",
    "Please portion the salmon such that my last bite is of the salmon. I would like to finish the broccoli and mash potatoes quickly. However, I can't take too much mash potatoes too quickly so alternate it with other foods. Serve me salmon only when I call for it or if there is no more of the other food.  ",
    "I would like to finish the cabbage quickly. I am ok with not finishing the rice. Only serve me the rice if you are to serve me pork cutlet in the next bit but not cabbage on the next bite. please ration the port cutlet so that I am left with a lot of pork cutlet at the end without any cabbage. ",
    "Please serve the teriyaki chicken in small portions while the rest in normal portions. Ensure that a bite of rice will not follow another bite of rice, instead alternate between the mixed vegetables and teriyaki chicken. Save at least one bite of teriyaki chicken at the end. Ensure I finish all the food unless I ask for all the teriyaki chicken or ask you to stop. "
]

ben_modified_interview_preferences = [
    "Please alternate rice with the other foods in any order. However, if the green beans, egg or fish is given in the bite before the rice, then do not repeat it.",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    ""
]

aaradh_interview_preferences = [
    "Give me mostly rice but also alternate between deep fried fish and green beans. Keep the tomato egg for the end. Give me enough time between bites so that I can chew properly.",
    "Start with only giving the fried chicken then mostly rice but also alternate between egg and green beans. ",
    "First only eggs then alternate between rice and fried chicken (more rice). Keep the cucumbers for the end.",
    "Alternate equally between rice, chicken and cucumber. ",
    "Start with just the steaks first, one piece at a time, but I will only bite half a piece at a time. So, the same piece should be given twice. Once steak is done, mashes potatoes. Then, finally alternate between green beans and carrots.",
    "Alternate between all (70% rice, 20% beef, 10% broccoli)",
    "Give me the meatballs first then alternate between green beans and mashed potatoes.",
    "Give me the salmon first then mashed potatoes then broccoli.",
    "Alternate between rice and pork cutlet (mostly rice) then at the end I want cabbage.",
    "Give me the teriyaki chicken first with rice (mostly teriyaki chicken) then once those are done, mixed vegetables."
]

# interview_preferences = [
#     "",
#     "",
#     "",
#     "",
#     "",
#     "",
#     "",
#     "",
#     "",
#     ""
# ]

interview_preferences = [
    "i dont want to eat any vege, and i only want to eat fish after i taste the rice.",
    "i dont like vege, just give me the rest.",
    "i want to finish the chicken first and eat the egg later with the rice, and i dont want to eat vege.",
    "give me the cucumber last, just feed me the rice and chicken randomly first.",
    "Feed me the steak, potato and vegetables in that order and repeat it. When feeding me vegetables please choose randomly.",
    "just give me everything randomly.",
    "just dont feed me the green bean.",
    "i dont like broccoli, but i like the potato. just feed me half of the potato and the whole of the fish.",
    "i dont want to eat the cabbage just give me the meat and the rice alternately.",
    "just feed me half of the vege alternately with the rest."
]

modified_interview_preferences = [
    "",
    "i dont like vege, just give me the rest. but don't feed me the same thing in a row",
    "",
    "",
    "Feed me steak, potato and a vegetable that you randomly choose. Repeat that order.",
    "",
    "",
    "",
    "",
    ""
]