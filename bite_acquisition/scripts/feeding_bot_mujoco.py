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

        self.item_portions = [2.4, 2.8, 1.2]
        
        mouth_pose = np.array([0.70, 0.0, 0.545])

        self.transfer_pose = PoseStamped()
        self.transfer_pose.pose.position.x = mouth_pose[0]
        self.transfer_pose.pose.position.y = mouth_pose[1]
        self.transfer_pose.pose.position.z = mouth_pose[2]

        self.execute = False

    def clear_plate(self):

        # user_preferences_hospital = [
        #     "I want to start with a spoonful of mashed potatoes, then have a small bite of chicken breast, and finish with a taste of steamed carrots.",
        #     "Alternate between bites of green beans and turkey slices, but make sure I get a spoonful of gravy-covered mashed potatoes every third bite.",
        #     "Serve me all the soup first, followed by alternating bites of plain rice and baked fish, and end with a nibble of bread roll.",
        #     "I like to mix bites of rice and steamed broccoli together, with a small slice of grilled chicken every fourth bite.",
        #     "Give me alternating bites of oatmeal and scrambled eggs for the first half, then finish with a couple of bites of fruit salad."
        # ]

        # bite_preferences_hospital = [
        #     "I want big spoonfuls of mashed potatoes, medium-sized bites of chicken, and very tiny bites of carrots.",
        #     "I’d like my turkey sliced thin and in small pieces, green beans in moderate bites, and mashed potatoes served in generous scoops.",
        #     "I prefer large chunks of baked fish, small bites of broccoli, and rice in medium spoonfuls.",
        #     "Serve the soup in big hearty spoonfuls, but keep the bread roll in small bite-sized pieces.",
        #     "I want small spoonfuls of oatmeal, large chunks of scrambled eggs, and fruit salad in tiny bites for a fresh finish."
        # ]

        # user_preferences_same_food = [
        #     # "Start with a big bite of rice mixed with chicken, followed by two small bites of egg, and finish with a pinch of plain rice.",
        #     "I want to eat alternating bites of egg and rice for a while, then switch to just chicken until the plate is almost empty.",
        #     "Serve me rice and egg together in the first few bites, then chicken in a standalone bite to reset my palate.",
        #     "I want small bites of rice with egg until there’s only half the rice left, then finish the plate with chicken-only bites.",
        #     "I like bites to alternate between chicken, rice, and egg in a clockwise rotation, but add an extra piece of chicken every fourth bite."
        # ]

        # bite_preferences_same_food = [
        #     "I want medium-sized bites of chicken, very tiny bites of egg, and rice served in generous spoonfuls.",
        #     "I’d like my chicken in large, hearty chunks, egg in delicate slivers, and rice in bite-sized scoops.",
        #     "Serve rice in tiny portions, egg in medium-sized bites, and chicken in large chunks with crispy edges.",
        #     "I prefer all my rice to be served in one big portion at the start, then small, even bites of chicken and egg for the rest.",
        #     "I want each bite to be a mix of chicken, egg, and rice, but the chicken should dominate in size and flavor."
        # ]

        # items = ['banana', 'chocolate sauce']
        # items = ['oatmeal', 'strawberry']
        # items = ['chicken', 'rice', 'egg']
        # items = ['red strawberry', 'chocolate sauce', 'ranch dressing', 'blue plate']
        # items = ['mashed potatoes']
        # items =  ['strawberry', 'ranch dressing', 'blue plate']

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

        user_preferences_hospital = [
            "I want to eat all the mashed potatoes first, then have a bite of turkey, and finish with some green beans.",
            "I like to alternate between bites of steamed carrots and grilled chicken, saving the rice for last.",
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
            "I’d like my soup served in hearty spoonfuls, bread in moderate pieces, and salad in small bites.",
            "I prefer big chunks of turkey, medium-sized bites of stuffing, and very tiny portions of cranberry sauce.",
            "Serve me mashed potatoes in generous portions, chicken in moderate bites, and peas in small spoonfuls.",
            "I want big bites of rice, small bites of steamed broccoli, and moderate bites of grilled fish.",
            "I like oatmeal in medium spoonfuls, scrambled eggs in larger portions, and toast in small, crisp pieces.",
            "Serve soup in big, warm spoonfuls, chicken in moderate bites, and vegetables in small, manageable portions.",
            "I’d like bread rolls served whole, mashed potatoes in large scoops, and meatloaf in thin slices.",
            "I prefer rice in medium-sized servings, steamed vegetables in small portions, and baked fish in larger pieces.",
            "I want sweet potatoes in generous servings, turkey in moderate portions, and mixed vegetables in smaller bites."
        ]



        # user_preference = "I want to eat alternating bites of chicken and rice most of the time but I would like to eat some egg occasionally."
        # bite_preference = "I want bigger bites of meat but smaller bites of rice."
        
    
        preference_idx = 0
        user_preference = user_preferences_hospital[preference_idx]
        bite_preference = bite_preferences_hospital[preference_idx]
        self.items = [food_items[preference_idx]]

        self.inference_server.FOOD_CLASSES = self.items

        bite_size = 0.0

        # Bite history
        bite_history = self.bite_history

        # Continue food
        continue_food_label = None

        # visualize = True

        actions_remaining = 10
        success = True
        
        while actions_remaining:

            print("--------------------")
            print('History', bite_history)
            print('Actions remaining', actions_remaining)
            print('Current user preference:', user_preference)
            print('Current bite preference:', bite_preference)
            ready = input('Ready?\n')
            if ready == 'n':
                exit(1)
            print("--------------------\n")

            # Get user preferences
            new_user_preference = input("Do you want to update your preference? Otherwise input [n] or Enter to continue\n")
            if new_user_preference not in ['n', '']:
                user_preference = new_user_preference
            print(f"USER PREFERENCE: {user_preference}\n")

            new_bite_preference = input("Do you want to update your bite size? Otherwise input [n] or Enter to continue\n")
            if new_bite_preference not in ['n', '']:
                bite_preference = new_bite_preference
            print(f"BITE PREFERENCE: {bite_preference}")
                
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
            
            
            food, bite_size = self.inference_server.get_autonomous_action(
                category_list, 
                labels_list, 
                per_food_portions, 
                user_preference, 
                bite_preference, 
                bite_size, 
                bite_history, 
                continue_food_label, 
                log_path
                )
            
            if food is None:
                exit(1)
            
            print(f"food: {food}\nbite_size: {bite_size}")
                
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
                    # self.item_portions[idx] -= 0.5
                    self.item_portions[idx] -= round(0.5 + (bite_size - -1.0) * (1.0 - 0.5) / (1.0 - -1.0), 2)
                    break
            bite_history.append((labels_list[food_id], bite_size))
            
            if success:
                actions_remaining -= 1

            with open('history.txt', 'w') as f:
                f.write(str(bite_history))

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
