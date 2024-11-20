# import cv2
import ast
import numpy as np
from scipy.spatial.transform import Rotation
import math
import os

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
        

    def clear_plate(self):
        # items = ['banana', 'chocolate sauce']
        # items = ['oatmeal', 'strawberry']
        items = ['chicken', 'rice', 'egg']
        # items = ['red strawberry', 'chocolate sauce', 'ranch dressing', 'blue plate']
        # items = ['mashed potatoes']
        # items =  ['strawberry', 'ranch dressing', 'blue plate']

        self.inference_server.FOOD_CLASSES = items

        user_preference = "I want to eat alternating bites of chicken and rice most of the time but I would like to eat some egg occasionally."
        # bite_preference = "I am fine with the amount I'm eating now."
        bite_preference = "I want bigger bites of meat but smaller bites of rice."
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
            print(f"USER PREFERENCE: {user_preference}")

            new_bite_preference = input("Do you want to update your bite size? Otherwise input [n] or Enter to continue\n")
            if new_bite_preference not in ['n', '']:
                bite_preference = new_bite_preference
            print(f"BITE PREFERENCE: {bite_preference}")
                
            log_path = self.log_file + str(self.log_count)
            self.log_count += 1

            # Hard coded for mujoco
            item_labels = ["chicken 0.82", "rice 0.76", "egg 0.84"]
            
            clean_item_labels = ["chicken", "rice", "egg"]

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
            
            
            food, bite_size = self.inference_server.get_autonomous_action(category_list, labels_list, per_food_portions, user_preference, bite_preference, bite_size, bite_history, continue_food_label, log_path)
            if food is None:
                exit(1)
            
            print(f"food: {food}\nbite_size: {bite_size}")
                
            print("--------------------")
            print(f"food_id: {food[0]} \naction_type: {food[1]} \nmetadata: {food[2]}")
            print("--------------------\n")

            food_id = food[0]
            action_type = food[1]
            metadata = food[2]
            
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
                    self.item_portions[idx] -= 0.5
                    break
            
            if success:
                actions_remaining -= 1

            with open('history.txt', 'w') as f:
                f.write(str(bite_history))

            k = input('Continue to transfer? Remember to start horizontal spoon.')
            while k not in ['y', 'n']:
                k = input('Continue to transfer? Remember to start horizontal spoon.')
            if k == 'y':
                self.skill_library.transfer_skill(self.transfer_pose)
                k = input('Continue to acquisition? Remember to shutdown horizontal spoon.')
                while k not in ['y', 'n']:
                    k = input('Continue to acquisition? Remember to shutdown horizontal spoon.\n')
                if k == 'n':
                    exit(1)

if __name__ == "__main__":

    args = None
    feeding_bot = FeedingBot()
    feeding_bot.clear_plate()
