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
        self.exit = False

        # Choose to use decomposer or not
        self.mode = 'no_decomposer'
        self.decomposer_output_directory = 'feeding_bot_output/test_outputs/decomposer_output/'
        self.no_decomposer_output_directory = 'feeding_bot_output/flair_tests/flair_output_v8/'

        if self.mode == 'decomposer':
            self.output_directory = self.decomposer_output_directory
            print('=== USING DECOMPOSER PROMPT ===')
        elif self.mode == 'no_decomposer':
            self.output_directory = self.no_decomposer_output_directory
            print('=== USING NON DECOMPOSER PROMPT ===')

    def clear_plate(self):

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
            "I have no preference in the sequence, but I prefer the spoon to be closer to me."
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

        for preference_idx in range(len(icorr_preferences)):

            user_preference = icorr_preferences[preference_idx]

            self.items = [icorr_food_items[preference_idx]]
            self.item_portions = [2.0, 2.0, 2.0]

            self.inference_server.FOOD_CLASSES = self.items

            # Bite history
            bite_history = []
            # Token history
            token_history = []

            # Continue food
            continue_food_label = None

            actions_remaining = 10
            success = True
            
            while actions_remaining:

                print(f"=== RUN NUMBER ===")
                print(preference_idx)
                print(f"=== ACTIONS REMAINING ===")
                print(actions_remaining)
                print(f"=== HISTORY ===")
                print(bite_history)
                print(f"=== USER PREFERENCE ===")
                print(user_preference)

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

                categories = self.inference_server.categorize_items(item_labels, sim=True) 

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
                print("Category List:", category_list)
                print("Labels List:", labels_list)
                print("Per Food Portions:", per_food_portions)
                print("--------------------\n")
                
                food, bite_size, distance_to_mouth, exit_angle, token_data = self.inference_server.get_autonomous_action(
                    category_list, 
                    labels_list, 
                    per_food_portions, 
                    user_preference, 
                    bite_history, 
                    preference_idx,
                    self.mode,
                    self.output_directory,
                    continue_food_label, 
                    log_path
                    )
                
                if food is not None:

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
                    
                    bite_history.append([labels_list[food_id], bite_size, distance_to_mouth, exit_angle])
                    
                    if success:
                        actions_remaining -= 1

                if actions_remaining == 0 or (food is None):
                    if self.mode == 'decomposer':
                        with open(self.output_directory + f'decomposer_output_idx_{preference_idx}.txt', 'a') as f:
                            f.write(f"=== FINAL HISTORY ===\n{bite_history}\n")
                            f.write(f"=== FINAL TOKEN HISTORY ===\n{token_history}\n")
                    elif self.mode == 'no_decomposer':
                        with open(self.output_directory + f'flair_output_idx_{preference_idx}.txt', 'a') as f:
                            f.write(f"=== FINAL HISTORY ===\n{bite_history}\n")
                            f.write(f"=== FINAL TOKEN HISTORY ===\n{token_history}\n")

                    with open(self.output_directory + f'histories_idx_{preference_idx}.txt', 'a') as f:
                        f.write(f"=== FINAL HISTORY ===\n{bite_history}\n")
                        f.write(f"=== FINAL TOKEN HISTORY ===\n{token_history}\n")

                    if food is None:
                        break

                if self.exit:
                    e = input("EXIT?")
                    if e in ['y', 'Y']:
                        exit(1)    

if __name__ == "__main__":

    args = None
    feeding_bot = FeedingBot()
    feeding_bot.clear_plate()
