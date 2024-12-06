# import cv2
import ast
import numpy as np
from scipy.spatial.transform import Rotation
import math
import os
import random

import rospy

from geometry_msgs.msg import PoseStamped

from skill_library_mujoco import SkillLibrary
from inference_class_mujoco import BiteAcquisitionInference

HOME_ORIENTATION = Rotation.from_quat([1/math.sqrt(2), 1/math.sqrt(2), 0, 0]).as_matrix()
DEFAULT_FORCE_THRESHOLD = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]

class FeedingBot:
    def __init__(self):

        rospy.init_node('FeedingBot')
        
        self.skill_library = SkillLibrary()

        print("Feeding Bot initialized\n")

        self.inference_server = BiteAcquisitionInference(mode='motion_primitive')

        if not os.path.exists("log"):
            os.mkdir("log")
        self.log_file = "log/"
        files = os.listdir(self.log_file)

        # if not os.path.exists('history.txt'):
        #     self.log_count = 1
        #     self.bite_history = []
        # else:
        #     with open('history.txt', 'r') as f:
        #         self.bite_history = ast.literal_eval(f.read().strip())
        #         self.log_count = len(self.bite_history)+1
        self.bite_history = []
        self.log_count = 1

        print("--------------------")
        print("Log count", self.log_count)
        print("History", self.bite_history)
        print("--------------------\n")
                
        # self.log_count should be the maximum numbered file in the log folder + 1
        self.log_count = max([int(x.split('_')[0]) for x in files]) + 1 if len(files) > 0 else 1
        
        mouth_pose = np.array([0.70, 0.0, 0.545])

        self.transfer_pose = PoseStamped()
        self.transfer_pose.pose.position.x = mouth_pose[0]
        self.transfer_pose.pose.position.y = mouth_pose[1]
        self.transfer_pose.pose.position.z = mouth_pose[2]

        self.execute = False
        self.preference_interrupt = False

        # Choose to use decomposer or not
        self.decompose = True
        self.old_prompt = False

    def clear_plate(self):

        food_items = [
            ["mashed potato", "turkey", "green bean"],
            ["carrot", "chicken", "rice"],
            ["soup", "bread", "salad"],
            ["turkey", "stuffing", "cranberry sauce"],
            ["rice", "broccoli", "grilled fish", "pudding"],
            ["oatmeal", "scrambled eggs", "toast"],
            ["soup", "chicken", "peas"],
            ["bread roll", "mashed potato", "meatloaf"],
            ["rice", "steamed vegetables", "baked fish"],
            ["turkey", "gravy", "sweet potato", "mixed vegetables"]
        ]

        user_preferences = [
            "I want to start with a generous serving of mashed potatoes, then move on to turkey slices, and finish with fresh green beans. Serve mashed potatoes in hearty portions, turkey in moderate bites, and green beans in smaller sizes. Keep the spoon close for mashed potatoes and tilt it gently upward when offering turkey.",

            "I prefer alternating bites of rice and grilled chicken, saving the steamed carrots for the end. Provide smaller bites of carrots, larger pieces of chicken, and medium scoops of rice. Hold the spoon slightly farther away for rice and tilt it downward for chicken.",

            "I’d like to begin with warm soup, then enjoy some soft bread, and finish with a small helping of salad. Offer the soup in large spoonfuls, bread in moderate pieces, and salad in delicate portions. Position the spoon at a medium range for soup and tilt it higher for salad.",

            "Serve me alternating bites of turkey and stuffing, with an occasional taste of cranberry sauce. Provide turkey in substantial cuts, stuffing in moderate scoops, and cranberry sauce in tiny nibbles. Hold the spoon close for cranberry sauce and angle it gently downward for turkey.",

            "I want rice and steamed broccoli served together, followed by a tender piece of grilled fish, and a sweet spoonful of pudding to finish. Serve rice and broccoli in small amounts, fish in larger portions, and pudding in medium, smooth servings. Keep the spoon at a mid-distance for rice and tilt it slightly upward for pudding.",

            "Let me start with oatmeal, move on to scrambled eggs, and end with a crispy bite of toast. Serve oatmeal in medium portions, eggs in larger bites, and toast in small, crisp pieces. Keep the spoon close for oatmeal and tilt it higher when offering toast.",

            "I’d like soup first, followed by a small piece of chicken, and finally a few peas. Serve the soup in big spoonfuls, chicken in medium chunks, and peas in tiny portions. Hold the spoon at a moderate distance for chicken and angle it slightly downward for peas.",

            "Begin with a warm bread roll, then mashed potatoes, and finish with a thin slice of meatloaf. Offer bread rolls whole, mashed potatoes in generous scoops, and meatloaf in manageable slices. Keep the spoon slightly farther back for meatloaf and tilt it upward for mashed potatoes.",

            "I enjoy alternating between spoonfuls of rice and steamed vegetables, with the occasional bite of baked fish. Provide rice and vegetables in smaller portions and baked fish in larger, tender pieces. Hold the spoon close for fish and angle it slightly downward for vegetables.",
            
            "I’d like to start with turkey and gravy, followed by creamy sweet potatoes, and finish with a light portion of mixed vegetables. Serve turkey in large portions, sweet potatoes in moderate servings, and mixed vegetables in small bites. Position the spoon at a medium range for sweet potatoes and tilt it gently upward for vegetables."
        ]  

        user_preferences_hospital = [
            "I want to eat all the mashed potatoes first, then have a bite of turkey, and finish with some green beans.",
            "I like to alternate between bites of rice and grilled chicken, saving the steamed carrots for last.",
            "I want to eat soup first, followed by bread, and end with small portions of salad.",
            "Serve me alternating bites of turkey and stuffing, with a nibble of cranberry sauce occasionally.",
            "I want to eat rice and steamed broccoli together, followed by grilled fish, and end with a taste of pudding.",
            "I like to start with oatmeal, then move to scrambled eggs, and finish with a bite of toast.",
            "I want a spoonful of soup first, then a small piece of chicken, and finally a few peas.",
            "Serve me bread rolls first, then mashed potatoes, and end with a small slice of meatloaf.",
            "I enjoy alternating between spoonfuls of rice and steamed vegetables, with an occasional bite of baked fish.",
            "I want to eat turkey and gravy first, followed by sweet potatoes, and finish with a small portion of mixed vegetables."
        ]

        bite_preferences_hospital = [
            "I want larger bites of meat and smaller bites of vegetables.",
            "Provide smaller bites of carrots, larger pieces of chicken, and medium scoops of rice.",
            "I prefer big chunks of turkey, medium-sized bites of stuffing, and very tiny portions of cranberry sauce.",
            "Serve me mashed potatoes in generous portions, chicken in moderate bites, and peas in small spoonfuls.",
            "I want big bites of rice, small bites of steamed broccoli, and moderate bites of grilled fish.",
            "I like oatmeal in medium spoonfuls, scrambled eggs in larger portions, and toast in small, crisp pieces.",
            "Serve soup in big, warm spoonfuls, chicken in moderate bites, and vegetables in small, manageable portions.",
            "I’d like bread rolls served whole, mashed potatoes in large scoops, and meatloaf in thin slices.",
            "I prefer rice in medium-sized servings, steamed vegetables in small portions, and baked fish in larger pieces.",
            "I want sweet potatoes in generous servings, turkey in moderate portions, and mixed vegetables in smaller bites."
        ]
    
        test_preferences_old = [
            "I want to eat the vegetables first, then the chicken, and save the rice for last.",
            "Start with a big piece of fish, followed by a small portion of salad.",
            "Serve me alternating bites of chicken and broccoli, with smaller bites of chicken",
            "I’d like rice served first, followed by vegetables.",
            "Finish the rice and meat first, then the soup.",
            "I prefer having smaller bites for rice but bigger bites for others",
            "I have no preference",
            "I want larger bites, and don't feed me any rice",
            "Serve me rice and chicken, with some vege in between",
            "Feed me potatoes last",

            "Keep the spoon close to my mouth when offering rice.",
            "Tilt the spoon slightly upward when feeding me soup.",
            "Hold the spoon farther away when offering meat.",
            "Position the spoon closer to my mouth.",
            "Tilt the spoon gently downward when serving vegetables.",

            "I’d like to start with soup and then move to rice. Feed closer for soup.",
            "Begin with mashed potatoes and finish with turkey. Move the spoon closer to me",
            "Serve alternating bites of rice and chicken. Hold the spoon farther for rice.",
            "Start with a medium bite of salad. Keep the spoon tilted higher for soup.",
            "I want to eat the chicken first. Tilt the spoon higher when feeding me."
            ]
        
        preferences = [
            "I want to eat the vegetables first, then the chicken, and save the rice for last.",
            "Start with a big piece of fish, followed by a small portion of salad.",
            "Serve me alternating bites of chicken and broccoli, with smaller bites of chicken",
            "I’d like rice served first, followed by vegetables.",
            "Finish the rice and meat first, then the soup.",
            "I prefer having smaller bites for rice but bigger bites for others",
            "I have no preference",
            "I want larger bites, and don't feed me any rice",
            "Serve me rice and chicken, with some vege in between",
            "Feed me potatoes last",

            "Keep the spoon close to my mouth when offering rice and make sure to scoop the rice slowly.",
            "Tilt the spoon slightly upward when feeding me soup and gently scoop the soup to prevent spillage.",
            "Hold the spoon farther away when offering meat and move a little slower when transferring the meat to my mouth.",
            "Position the spoon closer to my mouth.",
            "Tilt the spoon gently downward when serving vegetables.",

            "I’d like to start with soup and then move to rice. Feed closer for soup.",
            "Begin with mashed potatoes and finish with turkey. Move the spoon closer to me",
            "Serve alternating bites of rice and chicken. Hold the spoon farther for rice.",
            "Start with a medium bite of salad. Keep the spoon tilted higher for soup.",
            "I want to eat the chicken first. Tilt the spoon higher when feeding me."
            ]
        
        complex_preferences = [
            "I want to start with mashed potatoes and then move to chicken. Keep the spoon closer for mashed potatoes and tilt it downward for chicken.",
            "Begin with soup, followed by rice and vegetables. Feed the soup close to my mouth to prevent spilling.",
            "Serve alternating bites of rice and fish, saving vegetables for the end. Keep the spoon close to me.",
            "Start with small bites of salad, then move to larger bites of pasta. Tilt the spoon downward for salad and keep it farther for pasta.",
            "I’d like to eat chicken first, then rice, and finish with broccoli. Keep the spoon steady and level for rice and tilt it slightly upward for broccoli.",
            "Serve me mashed potatoes first, followed by turkey. Carefully scoop the mashed potatoes and tilt the spoon gently upward for turkey.",
            "Begin with small bites of soup, then alternate between rice and vegetables. Tilt the spoon upward for soup and transfer the vegetables with care to avoid spills.",
            "I want fish first, followed by small bites of salad. Hold the spoon at a closer range for fish and tilt it downward for salad.",
            "Start with medium bites of rice, followed by vegetables. I prefer the spoon to be further away from me, and bring the vegetables at a slower speed to avoid any mess.",
            "Begin with egg, then alternate between chicken and broccoli. Tilt the spoon upward for bread and keep it farther for broccoli.",
            "Serve the meat first, followed by rice. Keep the spoon close for meat and move the rice more gradually to prevent dropping any grains.",
            "Start with scrambled eggs, then meat. I prefer smaller bites.",
            "I only eat meat and vegetables, do not give me any rice. Move the food into my mouth during transfer.",
            "Do not feed me the meat, I want alternate bites always. Move the spoon close to me and do not tilt it.",
            "Begin with a medium bite of chicken, then rice. Tilt the spoon downward and keep it close for rice."
            ]


        complex_food_items = [
            ["mashed potatoes", "chicken", "peas"],
            ["soup", "rice", "carrots"],
            ["rice", "fish", "broccoli"],
            ["salad", "pasta", "potatoes"],
            ["chicken", "rice", "broccoli"],
            ["mashed potatoes", "turkey", "gravy"],
            ["soup", "rice", "carrots"],
            ["fish", "salad", "potatoes"],
            ["rice", "carrots", "peas"],
            ["egg", "chicken", "broccoli"],
            ["beef", "rice", "soup"],
            ["scrambled eggs", "beef", "toast"],
            ["beef", "carrots", "peas"],
            ["potatoes", "carrots", "peas"],
            ["chicken", "rice", "carrots"]
        ]

        
        food_items = [
            ["carrots", "chicken", "rice"],
            ["fish", "salad", "mashed potatoes"],
            ["chicken", "broccoli", "mashed potatoes"],
            ["rice", "carrots", "peas"],
            ["rice", "beef", "soup"],
            ["rice", "broccoli", "chicken"],
            ["mashed potato", "carrots", "beef"],
            ["peas", "rice", "meatballs"],
            ["rice", "chicken", "peas"],
            ["potatoes", "meatballs", "green beans"],

            ["rice", "beans", "corn"],
            ["soup", "rice", "chicken"],
            ["beef", "potatoes", "carrots"],
            ["yogurt", "fruit", "oatmeal"],
            ["carrots", "chicken", "mashed potatoes"],

            ["soup", "rice", "beef"],
            ["mashed potatoes", "turkey", "carrots"],
            ["rice", "chicken", "broccoli"],
            ["salad", "soup", "potatoes"],
            ["chicken", "rice", "carrots"]
        ]

        for preference_idx in range(6, 11): # len(preferences)):
            if not self.old_prompt:
                user_preference = complex_preferences[preference_idx]
            else:
                user_preference = user_preferences_hospital[preference_idx]
                bite_preference = bite_preferences_hospital[preference_idx]
                transfer_preference = "Hold the spoon slightly farther away for rice and tilt it downward for chicken."

            self.items = [complex_food_items[preference_idx]]
            self.item_portions = [2.0, 2.0, 2.0]

            self.inference_server.FOOD_CLASSES = self.items

            # Bite history
            bite_history = []
            # Token history
            token_history = []

            # Continue food
            continue_food_label = None

            # visualize = True

            actions_remaining = 10
            success = True
            
            while actions_remaining:

                print(f"=== RUN NUMBER ===")
                print(preference_idx)
                print(f"=== RUN NUMBER ===")

                print("--------------------")
                print('History', bite_history)
                print('Token History', token_history)
                print('Actions remaining', actions_remaining)
                print('Current user preference:', user_preference)
                # ready = input('Ready?\n')
                # if ready == 'n':
                #     exit(1)
                print("--------------------\n")

                if self.preference_interrupt:
                    # Get user preferences
                    print(f"CURRENT USER PREFERENCE: {user_preference}")
                    new_user_preference = input("Do you want to update your preference? Otherwise input [n] or Enter to continue\n")
                    if new_user_preference not in ['n', '']:
                        user_preference = new_user_preference
                        print(f"NEW USER PREFERENCE: {user_preference}\n")

                    print(f"CURRENT BITE PREFERENCE: {bite_preference}")
                    new_bite_preference = input("Do you want to update your bite size? Otherwise input [n] or Enter to continue\n")
                    if new_bite_preference not in ['n', '']:
                        bite_preference = new_bite_preference
                        print(f"NEW BITE PREFERENCE: {bite_preference}\n")
                    
                    print(f"CURRENT DISTANCE TO MOUTH PREFERENCE: {distance_to_mouth_preference}")
                    new_distance_to_mouth_preference = input("Do you want to update your distance to mouth preference? Otherwise input [n] or Enter to continue\n")
                    if new_distance_to_mouth_preference not in ['n', '']:
                        distance_to_mouth_preference = new_distance_to_mouth_preference
                        print(f"NEW DISTANCE TO MOUTH PREFERENCE: {distance_to_mouth_preference}\n")
                        
                    print(f"CURRENT EXIT ANGLE PREFERENCE: {exit_angle_preference}")
                    new_exit_angle_preference = input("Do you want to update your exit angle preference? Otherwise input [n] or Enter to continue\n")
                    if new_exit_angle_preference not in ['n', '']:
                        exit_angle_preference = new_exit_angle_preference
                        print(f"NEW EXIT ANGLE PREFERENCE: {exit_angle_preference}\n")
                    
                log_path = self.log_file + str(self.log_count)
                self.log_count += 1

                # Hard coded for mujoco
                food_item_labels = [[f"{food} {random.uniform(0.5, 1.0):.2f}" for food in items] for items in self.items]
                item_labels = food_item_labels[0]
                
                clean_item_labels = self.items[0] 

                categories = self.inference_server.categorize_items(item_labels, sim=False) 

                print("--------------------")
                print("Labels:", item_labels)
                print("Categories:", categories)
                print("Portions:", self.item_portions)        
                print("--------------------\n")

                category_list = []
                labels_list = []
                per_food_portions = []

                for i in range(len(categories)):
                    if labels_list.count(clean_item_labels[i]) == 0:
                        category_list.append(categories[i])
                        labels_list.append(clean_item_labels[i])
                        per_food_portions.append(self.item_portions[i])
                    else:
                        index = labels_list.index(clean_item_labels[i])
                        per_food_portions[index] += self.item_portions[i]

                print("--------------------")
                print("Bite History", bite_history)
                print("Category List:", category_list)
                print("Labels List:", labels_list)
                print("Per Food Portions:", per_food_portions)
                print("--------------------\n")
                
                if self.decompose:
                    print("USING DECOMPOSER")
                    food, bite_size, speed_of_acquisition, distance_to_mouth, exit_angle, speed_of_transfer, token_data = self.inference_server.get_autonomous_action_decomposer(
                        category_list, 
                        labels_list, 
                        per_food_portions, 
                        user_preference, 
                        bite_history, 
                        preference_idx,
                        continue_food_label, 
                        log_path
                        )
                elif self.old_prompt:
                    print("USING OLD PROMPT")
                    food, bite_size, distance_to_mouth, exit_angle = self.inference_server.get_autonomous_action_old_prompt(
                        category_list, 
                        labels_list, 
                        per_food_portions, 
                        user_preference, 
                        bite_preference, 
                        transfer_preference,
                        bite_history, 
                        continue_food_label, 
                        log_path
                        )
                else:
                    print("USING NON DECOMPOSER")
                    food, bite_size, speed_of_acquisition, distance_to_mouth, exit_angle, speed_of_transfer, token_data = self.inference_server.get_autonomous_action_no_decomposer(
                    category_list, 
                    labels_list, 
                    per_food_portions, 
                    user_preference, 
                    bite_history, 
                    preference_idx,
                    continue_food_label, 
                    log_path
                    )
                
                if food is None:
                    exit(1)
                
                print("\n--------------------")
                print(f"food: {food}")
                print(f"bite: {labels_list[food[0]]}")
                print(f"bite_size: {bite_size}")
                print(f"distance_to_mouth: {distance_to_mouth}")
                print(f"entry_angle: {exit_angle}")
                print("--------------------\n")
                    
                print("--------------------")
                print(f"food_id: {food[0]} \naction_type: {food[1]} \nmetadata: {food[2]}")
                print("--------------------\n")

                food_id = food[0]
                action_type = food[1]
                metadata = food[2]

                if self.execute:
                
                    if action_type == 'Scoop':
                        scooping_point = metadata['scooping_point']
                        action = self.skill_library.scooping_skill_mujoco(keypoints = scooping_point, bite_size = bite_size)

                    elif action_type == 'Push':
                        continue_food_label = labels_list[food_id]
                        start, end = metadata['start'], metadata['end']
                        input('Continue pushing skill?')
                        action = self.skill_library.pushing_skill_mujoco(keypoints = [start, end])

                    elif action_type == 'Cut':
                        continue_food_label = labels_list[food_id]
                        cut_point = metadata['point']
                        cut_angle = metadata['cut_angle']
                        action = self.skill_library.cutting_skill_mujoco(keypoint = cut_point, cutting_angle = cut_angle)            

                    if action_type == 'Scoop': # Terminal actions
                        continue_food_label = None
                        bite_history.append((labels_list[food_id], bite_size))

                for idx in range(len(clean_item_labels)):
                    if clean_item_labels[food_id] == clean_item_labels[idx]:
                        self.item_portions[idx] -= 0.45
                        # self.item_portions[idx] -= round(0.5 + (bite_size - -1.0) * (1.0 - 0.5) / (1.0 - -1.0), 2)
                        break
                
                bite_history.append([labels_list[food_id], bite_size, speed_of_acquisition, distance_to_mouth, exit_angle, speed_of_transfer])
                
                if not self.old_prompt:
                    token_history.append(token_data)
                if success:
                    actions_remaining -= 1

                with open('token_history.txt', 'w') as f:
                    f.write(str(token_history))

                print("=== ACTIONS REMAINING ===")
                print(actions_remaining)
                print("=== ACTIONS REMAINING ===\n")

                if actions_remaining == 0:
                    if self.decompose:
                        with open(f'feeding_bot_output/icorr_outputs_v3/decomposer_output/decomposer_output_idx_{preference_idx}.txt', 'a') as f:
                            f.write(f"=== FINAL HISTORY ===\n{bite_history}\n")
                            f.write(f"=== FINAL TOKEN HISTORY ===\n{token_history}\n")
                        with open(f'feeding_bot_output/icorr_outputs_v3/decomposer_output/histories_idx_{preference_idx}.txt', 'a') as f:
                            f.write(f"=== FINAL HISTORY ===\n{bite_history}\n")
                            f.write(f"=== FINAL TOKEN HISTORY ===\n{token_history}\n")
                    else:
                        with open(f'feeding_bot_output/icorr_outputs_v3/no_decomposer_output/no_decomposer_output_idx_{preference_idx}.txt', 'a') as f:
                            f.write(f"=== FINAL HISTORY ===\n{bite_history}\n")
                            f.write(f"=== FINAL TOKEN HISTORY ===\n{token_history}\n")
                        with open(f'feeding_bot_output/icorr_outputs_v3/no_decomposer_output/histories_idx_{preference_idx}.txt', 'a') as f:
                            f.write(f"=== FINAL HISTORY ===\n{bite_history}\n")
                            f.write(f"=== FINAL TOKEN HISTORY ===\n{token_history}\n")
                    bite_history = []

if __name__ == "__main__":

    args = None
    feeding_bot = FeedingBot()
    feeding_bot.clear_plate()
