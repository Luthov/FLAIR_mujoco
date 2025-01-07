# import cv2
import time
import os
import numpy as np

from preference_planner import PreferencePlanner

import os
from openai import OpenAI
import ast
import sys

import base64
import requests

PATH_TO_GROUNDED_SAM = '/home/rkjenamani/Grounded-Segment-Anything'
PATH_TO_DEPTH_ANYTHING = '/home/rkjenamani/Depth-Anything'
PATH_TO_SPAGHETTI_CHECKPOINTS = '/home/rkjenamani/flair_ws/src/bite_acquisition/spaghetti_checkpoints'
USE_EFFICIENT_SAM = False

sys.path.append(PATH_TO_DEPTH_ANYTHING)

import random

class GPT4VFoodIdentification:
    def __init__(self, api_key, prompt_dir):

        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
            }
        self.prompt_dir = prompt_dir
        
        with open("%s/prompt.txt"%self.prompt_dir, 'r') as f:
            self.prompt_text = f.read()
        
        self.detection_prompt_img1 = cv2.imread("%s/11.jpg"%self.prompt_dir)
        self.detection_prompt_img2 = cv2.imread("%s/12.jpg"%self.prompt_dir)
        self.detection_prompt_img3 = cv2.imread("%s/13.jpg"%self.prompt_dir)

        self.detection_prompt_img1 = self.encode_image(self.detection_prompt_img1)
        self.detection_prompt_img2 = self.encode_image(self.detection_prompt_img2)
        self.detection_prompt_img3 = self.encode_image(self.detection_prompt_img3)

        self.mode = 'ours' # ['ours', 'preference', 'efficiency']

    def encode_image(self, openCV_image):
        retval, buffer = cv2.imencode('.jpg', openCV_image)
        return base64.b64encode(buffer).decode('utf-8')

    def prompt_zero_shot(self, image, prompt):
        # Getting the base64 string
        base64_image = self.encode_image(image)

        payload = {
        "model": "gpt-4o",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        response_text =  response.json()['choices'][0]["message"]["content"]
        return response_text
        
    def prompt(self, image):
        
        # Getting the base64 string
        base64_image = self.encode_image(image)

        payload = {
        "model": "gpt-4o",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": self.prompt_text
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.detection_prompt_img1}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.detection_prompt_img2}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.detection_prompt_img3}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        response_text =  response.json()['choices'][0]["message"]["content"]

        return response_text

class BiteAcquisitionInference:
    def __init__(self, mode):

        # GroundingDINO config and checkpoint
        self.GROUNDING_DINO_CONFIG_PATH = PATH_TO_GROUNDED_SAM + "/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.GROUNDING_DINO_CHECKPOINT_PATH = PATH_TO_GROUNDED_SAM + "/groundingdino_swint_ogc.pth"
        
        self.use_efficient_sam = USE_EFFICIENT_SAM

        self.FOOD_CLASSES = ["spaghetti", "meatball"]
        self.BOX_THRESHOLD = 0.3
        self.TEXT_THRESHOLD = 0.2
        self.NMS_THRESHOLD = 0.4

        self.CATEGORIES = ['meat/seafood', 'vegetable', 'noodles', 'fruit', 'dip', 'plate']

        # read API key from command line argument
        self.api_key =  os.environ['OPENAI_API_KEY']

        self.client = OpenAI(api_key=self.api_key)

        # torch.set_flush_denormal(True)
        checkpoint_dir = PATH_TO_SPAGHETTI_CHECKPOINTS

        self.preference_planner = PreferencePlanner()

        self.mode = mode

    def recognize_items(self, image):
        response = self.gpt4v_client.prompt(image).strip()
        items = ast.literal_eval(response)
        return items

    def chat_with_openai(self, prompt):
        """
        Sends the prompt to OpenAI API using the chat interface and gets the model's response.
        """
        message = {
                    'role': 'user',
                    'content': prompt
                  }
    
        response = self.client.chat.completions.create(
                   model='gpt-3.5-turbo-1106',
                   messages=[message]
                  )
        
        # Extract the chatbot's message from the response.
        # Assuming there's at least one response and taking the last one as the chatbot's reply.
        chatbot_response = response.choices[0].message.content
        return chatbot_response.strip()

    def determine_action(self, density, entropy, valid_actions):
        
        DENSITY_THRESH = 0.64 # 0.8 for mashed potatoes
        ENTROPY_THRESH = 9.0
        if density > DENSITY_THRESH:
            if 'Acquire' in valid_actions:
                return 'Acquire'
            elif 'Push Filling' in valid_actions:
                return 'Push Filling'
        elif entropy > ENTROPY_THRESH:
            if 'Group' in valid_actions:
                return 'Group'
            elif 'Push Filling' in valid_actions:
                return 'Push Filling'
        elif 'Push Filling' in valid_actions:
            return 'Push Filling'
        return 'Acquire'
    
    def get_scoop_action_mujoco(self, food_label):
        action = "Acquire"
        # Placeholder for scoop keypoints
        if food_label == 'rice':
            scoop_keypoints = [[0.4, -0.25, 0.075]]
        elif food_label == 'chicken':
            scoop_keypoints = [[0.4, 0, 0.075]]
        # elif food_label == 'egg':
        scoop_keypoints = [[0.4, 0.25, 0.075]]
        # scoop_keypoints = [[0.3507, 0.0512, 0.0373 + 0.1]]

        push_keypoints = [[0.3507, 0.0512, 0.0373 + 0.1], [0.4507, -0.0512, 0.0373 + 0.1]]
        return action, scoop_keypoints[0], push_keypoints[0], push_keypoints[1]

    def clean_labels(self, labels):
        clean_labels = []
        instance_count = {}
        for label in labels:
            label = label[:-4].strip()
            clean_labels.append(label)
            if label in instance_count:
                instance_count[label] += 1
            else:
                instance_count[label] = 1
        return clean_labels, instance_count

    def categorize_items(self, labels, sim=True):
        categories = []

        if sim:
            for label in labels:
                if 'noodle' in label or 'fettuccine' in label:
                    categories.append('noodles')
                elif 'mashed' in label or 'oatmeal' in label:
                    categories.append('semisolid')
                elif 'banana' in label or 'strawberry' in label or 'watermelon' in label or 'celery' in label or 'baby carrot' in label:
                    categories.append('fruit')
                elif 'broccoli' in label:
                    categories.append('vegetable')
                elif 'blue' in label:
                    categories.append('plate')
                elif 'sausage' in label or 'meatball' in label or 'meat' in label or 'chicken' in label:
                    categories.append('meat/seafood')
                elif 'brownie' in label:
                    categories.append('brownie')
                elif 'ranch dressing' in label or 'ketchup' in label or 'caramel' in label or 'chocolate sauce' in label:
                    categories.append('dip')
                elif 'rice' in label:
                    categories.append('semisolid')
                elif 'chicken' in label:
                    categories.append('meat/seafood')
                elif 'carrot' in label:
                    categories.append('vegetable')
                elif 'salad' in label:
                    categories.append('vegetable')
                elif 'pea' in label:
                    categories.append('vegetable')
                elif 'beef' in label:
                    categories.append('meat/seafood')
                elif 'spinach' in label:
                    categories.append('vegetable')
                elif 'broccoli' in label:
                    categories.append('vegetable')
                elif 'corn' in label:
                    categories.append('vegetable')
                elif 'cabbage' in label:
                    categories.append('vegetable')
                elif 'potato' in label:
                    categories.append('vegetable')
                else:
                    raise KeyError(f"Label {label} not recognized")

        else:
            prompt = """
                    Acceptable outputs: ['noodles', 'meat/seafood', 'vegetable', 'brownie', 'dip', 'fruit', 'plate', 'semisolid']

                    Input: 'noodles 0.69'
                    Output: 'noodles'

                    Input: 'shrimp 0.26'
                    Output: 'meat/seafood'

                    Input: 'meat 0.46'
                    Output: 'meat/seafood'

                    Input: 'broccoli 0.42'
                    Output: 'vegetable'

                    Input: 'celery 0.69'
                    Output: 'vegetable'

                    Input: 'baby carrot 0.47'
                    Output: 'vegetable'

                    Input: 'chicken 0.27'
                    Output: 'meat/seafood'

                    Input: 'brownie 0.47'
                    Output: 'brownie'

                    Input: 'ketchup 0.47'
                    Output: 'dip'

                    Input: 'ranch 0.24'
                    Output: 'dip'

                    Input: 'mashed potato 0.43'
                    Output: 'semisolid'

                    Input: 'mashed potato 0.30'
                    Output: 'semisolid'

                    Input: 'risotto 0.40'
                    Output: 'semisolid'

                    Input: 'oatmeal 0.43'
                    Output: 'semisolid'

                    Input: 'caramel 0.28'
                    Output: 'dip'

                    Input: 'chocolate sauce 0.24'
                    Output: 'dip'

                    Input: 'strawberry 0.57'
                    Output: 'fruit'

                    Input: 'watermelon 0.47'
                    Output: 'fruit'

                    Input: 'oatmeal 0.43'
                    Output: 'semisolid'

                    Input: 'salad 0.47'
                    Output: 'vegetable'

                    Input: 'blue'
                    Output: 'plate'

                    Input: 'blue'
                    Output: 'plate'

                    Input: 'blue plate'
                    Output: 'plate'

                    Input: 'blue plate'
                    Output: 'plate'

                    Input: 'blueberry 0.87'
                    Output: 'fruit'

                    Input: 'rice 0.50'
                    Output: 'semisolid'

                    Input: '%s'
                    Output:
                    """
            for label in labels:
                predicted_category = self.chat_with_openai(prompt%label).strip().replace("'",'')
                categories.append(predicted_category)

        return categories
    
    def get_autonomous_action(self, 
                              categories, 
                              labels, 
                              portions, 
                              preference,
                              history,
                              preference_idx,
                              mode, 
                              output_directory,
                              continue_food_label = None, 
                              log_path = None):
        
        if continue_food_label is not None:
            food_to_consider = [i for i in range(len(labels)) if labels[i] == continue_food_label]
        else:
            food_to_consider = range(len(categories))

        next_actions = []
        efficiency_scores = []        

        for idx in food_to_consider:

            if categories[idx] == 'semisolid':                
                action, scooping_point, filling_push_start, filling_push_end = self.get_scoop_action_mujoco(labels[idx])

                if action == 'Acquire':
                    efficiency_scores.append(1)
                    next_actions.append((idx, 'Scoop', {'scooping_point': scooping_point}))

                else:
                    efficiency_scores.append(2)
                    next_actions.append((idx, 'Push', {'start':filling_push_start, 'end':filling_push_end}))

            elif categories[idx] in ['meat/seafood', 'vegetable', 'fruit', 'brownie']:
                action, scooping_point, filling_push_start, filling_push_end = self.get_scoop_action_mujoco(labels[idx])
                efficiency_scores.append(1)
                next_actions.append((idx, 'Scoop', {'scooping_point': scooping_point}))
        
        print('Length of next actions: ', len(next_actions))
        if self.mode == 'efficiency':
            return next_actions[np.argmin(efficiency_scores)], None
        
        # round efficiency scores to nearest integer
        efficiency_scores = [round(score) for score in efficiency_scores]

        # take reciprocal of efficiency scores and multiply with LCM
        print('Efficiency scores before reciprocal: ', efficiency_scores)
        efficiency_scores = np.array([1/score for score in efficiency_scores]) * int(np.lcm.reduce(efficiency_scores))
        efficiency_scores = efficiency_scores.astype(int).tolist()

        non_dip_labels = []
        non_dip_portions_rounded = []
        for idx in range(len(labels)):
            non_dip_labels.append(labels[idx])
            non_dip_portions_rounded.append(round(portions[idx]))

        if continue_food_label is not None:
            next_bite = [continue_food_label]
        else:
            print('Non dip labels: ', non_dip_labels)
            print('Efficiency scores: ', efficiency_scores)
            print('Bite portions: ', non_dip_portions_rounded)

        print("=== CALLING PLANNER ===")

        next_bite, bite_size, distance_to_mouth, exit_angle, transfer_speed, token_data = self.preference_planner.plan(
            non_dip_labels, 
            non_dip_portions_rounded, 
            efficiency_scores, 
            preference, 
            history,
            preference_idx,
            mode,
            output_directory
            )

        print(non_dip_labels, next_bite)
    
        if next_bite == [] or next_bite == '':
            return  None, None, None, None, None, None
        else:
            idx = non_dip_labels.index(next_bite)
            return next_actions[idx], bite_size, distance_to_mouth, exit_angle, transfer_speed, token_data