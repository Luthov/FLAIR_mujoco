import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preference_planner import PreferencePlanner
import ast

planner = PreferencePlanner()

# user_preference = "I want to start with oatmeal in medium spoonfuls, then move to scrambled eggs in larger poritons, and finish with a bite of toast in small, crisp pieces. Keep the spoon closer to my mouth when feeding me oatmeal and also tilt the spoon higher as the spoon is exiting my mouth"

user_preferences = [
    # "I want to start with a generous serving of mashed potatoes, then move on to turkey slices, and finish with fresh green beans. Serve mashed potatoes in hearty portions, turkey in moderate bites, and green beans in smaller sizes. Keep the spoon close for mashed potatoes and tilt it gently upward when offering turkey."

    "I prefer alternating bites of rice and grilled chicken, saving the steamed carrots for the end. Provide smaller bites of carrots, larger pieces of chicken, and medium scoops of rice. Hold the spoon slightly farther away for carrots and tilt it downward for chicken."

    # "I’d like to begin with warm soup, then enjoy some soft bread, and finish with a small helping of salad. Offer the soup in large spoonfuls, bread in moderate pieces, and salad in delicate portions. Position the spoon at a medium range for soup and tilt it higher for salad.",

    # "Serve me alternating bites of turkey and stuffing, with an occasional taste of cranberry sauce. Provide turkey in substantial cuts, stuffing in moderate scoops, and cranberry sauce in tiny nibbles. Hold the spoon close for cranberry sauce and angle it gently downward for turkey.",

    # "I want rice and steamed broccoli served together, followed by a tender piece of grilled fish, and a sweet spoonful of pudding to finish. Serve rice and broccoli in small amounts, fish in larger portions, and pudding in medium, smooth servings. Keep the spoon at a mid-distance for rice and tilt it slightly upward for pudding.",

    # "Let me start with oatmeal, move on to scrambled eggs, and end with a crispy bite of toast. Serve oatmeal in medium portions, eggs in larger bites, and toast in small, crisp pieces. Keep the spoon close for oatmeal and tilt it higher when offering toast.",

    # "I’d like soup first, followed by a small piece of chicken, and finally a few peas. Serve the soup in big spoonfuls, chicken in medium chunks, and peas in tiny portions. Hold the spoon at a moderate distance for chicken and angle it slightly downward for peas.",

    # "Begin with a warm bread roll, then mashed potatoes, and finish with a thin slice of meatloaf. Offer bread rolls whole, mashed potatoes in generous scoops, and meatloaf in manageable slices. Keep the spoon slightly farther back for meatloaf and tilt it upward for mashed potatoes.",

    # "I enjoy alternating between spoonfuls of rice and steamed vegetables, with the occasional bite of baked fish. Provide rice and vegetables in smaller portions and baked fish in larger, tender pieces. Hold the spoon close for fish and angle it slightly downward for vegetables.",
    
    # "I’d like to start with turkey and gravy, followed by creamy sweet potatoes, and finish with a light portion of mixed vegetables. Serve turkey in large portions, sweet potatoes in moderate servings, and mixed vegetables in small bites. Position the spoon at a medium range for sweet potatoes and tilt it gently upward for vegetables."
]
items = ['carrot', 'chicken', 'rice']
efficiency_scores =  [1, 1, 1]
bite_portions = [5, 5, 5]
history = [['rice', 0.0, 7.0, 95.0], ['chicken', 0.5, 7.5, 90.0], ['rice', 0.0, 7.0, 95.0]]

planner.plan_motion_and_transfer_primitives(items, bite_portions, efficiency_scores, user_preferences[0], history)

# preference_list = []
# for preference in user_preferences:
#     bite_preference, transfer_preference = planner.parse_preferences(preference)
#     preference_list.append((bite_preference, transfer_preference))

#     with open("preference_list.txt", "a") as f:  # Changed "w" to "a" to append to the file
#         f.write(f"USER PREFERENCE {preference}\n")
#         f.write(f"BITE PREFERENCE: {bite_preference}\n")
#         f.write(f"TRANSFER PREFERENCE: {transfer_preference}\n\n")