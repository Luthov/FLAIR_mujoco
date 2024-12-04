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

        if not os.path.exists('history.txt'):
            self.log_count = 1
            self.bite_history = []
        else:
            with open('history.txt', 'r') as f:
                self.bite_history = ast.literal_eval(f.read().strip())
                self.log_count = len(self.bite_history)+1

        print("--------------------")
        print("Log count", self.log_count)
        print("History", self.bite_history)
        print("--------------------\n")
                
        # self.log_count should be the maximum numbered file in the log folder + 1
        self.log_count = max([int(x.split('_')[0]) for x in files]) + 1 if len(files) > 0 else 1

        self.item_portions = [5.0, 5.0, 5.0]
        
        mouth_pose = np.array([0.70, 0.0, 0.545])

        self.transfer_pose = PoseStamped()
        self.transfer_pose.pose.position.x = mouth_pose[0]
        self.transfer_pose.pose.position.y = mouth_pose[1]
        self.transfer_pose.pose.position.z = mouth_pose[2]

        self.execute = False
        self.preference_interrupt = False

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
    
        preference_idx = 1
        user_preference = user_preferences[preference_idx]
        self.items = [food_items[preference_idx]]

        self.inference_server.FOOD_CLASSES = self.items

        bite_size = 0.0
        distance_to_mouth = 7.5
        entry_angle = 90.0
        exit_angle = 90.0

        # Bite history
        bite_history = self.bite_history

        # Token history
        token_history = []

        # Continue food
        continue_food_label = None

        # visualize = True

        actions_remaining = 10
        success = True
        
        while actions_remaining:

            print("--------------------")
            print('History', bite_history)
            print('Token History', token_history)
            print('Actions remaining', actions_remaining)
            print('Current user preference:', user_preference)
            ready = input('Ready?\n')
            if ready == 'n':
                exit(1)
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
            # item_labels = ["chicken 0.82", "rice 0.76", "egg 0.84"]
            
            clean_item_labels = self.items[0] # ["chicken", "rice", "egg"]
            # print(f"Clean item labels: {clean_item_labels}")

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
            
            
            # food, bite_size, distance_to_mouth, exit_angle = self.inference_server.get_autonomous_action(
            #     category_list, 
            #     labels_list, 
            #     per_food_portions, 
            #     user_preference, 
            #     bite_preference, 
            #     distance_to_mouth_preference,
            #     exit_angle_preference,
            #     bite_size, 
            #     bite_history, 
            #     continue_food_label, 
            #     log_path
            #     )

            food, bite_size, distance_to_mouth, exit_angle, token_data = self.inference_server.get_autonomous_action_test(
                category_list, 
                labels_list, 
                per_food_portions, 
                user_preference, 
                bite_history, 
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
                    self.item_portions[idx] -= 0.2
                    # self.item_portions[idx] -= round(0.5 + (bite_size - -1.0) * (1.0 - 0.5) / (1.0 - -1.0), 2)
                    break
            bite_history.append([labels_list[food_id], bite_size, distance_to_mouth, exit_angle])
            token_history.append(token_data)
            if success:
                actions_remaining -= 1

            with open('history.txt', 'w') as f:
                f.write(str(bite_history))

            with open('token_history.txt', 'w') as f:
                f.write(str(token_history))

            k = input('Exit?')
            if k == 'y':
                exit(1)
            # k = input('Continue to transfer? Remember to start horizontal spoon.')
            # while k not in ['y', 'n']:
            #     k = input('Continue to transfer? Remember to start horizontal spoon.')
            # if k == 'y':
            #     self.skill_library.transfer_skill(self.transfer_pose)
            #     k = input('Continue to acquisition? Remember to shutdown horizontal spoon.')
            #     while k not in ['y', 'n']:
            #         k = input('Continue to acquisition? Remember to shutdown horizontal spoon.\n')
            #     if k == 'n':
            #         exit(1)

if __name__ == "__main__":

    args = None
    feeding_bot = FeedingBot()
    feeding_bot.clear_plate()
