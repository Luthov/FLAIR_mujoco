# import cv2
import time
import numpy as np
import math
import os

import base64
import requests
import json

from openai import OpenAI
import ast

class GPTInterface:
    def __init__(self):
        self.api_key =  os.environ.get('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        
    def chat_with_openai(self, prompt):
        """
        Sends the prompt to OpenAI API using the chat interface and gets the model's response.
        """
        message = {
                    'role': 'user',
                    'content': prompt
                  }
        response = self.client.chat.completions.create(
                   model='gpt-4o-2024-08-06',
                #    model='gpt-4o-mini-2024-07-18',
                #    model='gpt-4-turbo-2024-04-09', # 'gpt-4-0125-preview', 
                   messages=[message]
                  )
        # print(response)
        chatbot_response = response.choices[0].message.content

        chatbot_usage = response.usage
        tokens = [chatbot_usage.completion_tokens, chatbot_usage.prompt_tokens, chatbot_usage.total_tokens]

        return chatbot_response.strip(), tokens

class PreferencePlanner:
    def __init__(self):
        self.gpt_interface = GPTInterface()

        self.decomposer_prompt_file = 'decomposer_prompts/decomposer.txt'
        self.bite_sequencing_prompt_file = 'decomposer_prompts/bite_acquisition_flair.txt'
        self.transfer_parameter_prompt_file = 'decomposer_prompts/bite_transfer.txt'

        # self.no_decomposer_prompt_file = 'flair_testing/flair_v9.txt'
        self.no_decomposer_prompt_file = 'improved_prompt_v6.txt'

        self.debug = False

        self.update_bite_preference = False
        self.update_transfer_params = False
        self.current_motion_params = 'None'


    def parse_preferences(self, preference):
        
        with open('prompts/' + self.decomposer_prompt_file, 'r') as f:
            prompt = f.read()

        prompt = prompt%(preference)

        response, decomposer_tokens = self.gpt_interface.chat_with_openai(prompt)

        intermediate_response = response.split('Feeding sequence preference: ')[1].strip()
        preferences = intermediate_response.split('\n')
        bite_preference = preferences[0]
        for pref in preferences:
            if 'Motion parameters preference: ' in pref:
                motion_parameter_preference = pref.split('Motion parameters preference: ')[1].strip()

        print(f"=== USER PREFERENCE ===")
        print(preference)
        print(f"=== BITE PREFERENCE ===")
        print(bite_preference)
        print(f"=== TRANSFER PREFERENCE ===")
        print(motion_parameter_preference)

        if bite_preference != 'None':
            self.bite_preference = bite_preference
            self.update_bite_preference = True

        if motion_parameter_preference != 'None':
            self.transfer_preference = motion_parameter_preference
            self.update_transfer_params = True
        
        return decomposer_tokens
    
    def plan(self, 
             items, 
             portions, 
             preference, 
             history, 
             preference_change, 
             preference_idx, 
             mode,
             output_directory):

        portions_sentence = str(portions)
        print('\n==== ITEMS / PORTIONS REMAINING ===')
        print(f"{items} / {portions_sentence}")
    
        if mode == 'decomposer':

            # Extracting bite preference and transfer preference
            if preference_change:
                _ = self.parse_preferences(preference)

            # Reading prompts
            with open('prompts/' + self.bite_sequencing_prompt_file, 'r') as f:
                bite_sequencing_prompt = f.read()

            # Extracting bite sequencing history and transfer parameters history
            bite_sequencing_history = [item[:1] for item in history]

            bite_sequencing_prompt = bite_sequencing_prompt%(
                str(items), 
                portions_sentence,
                str(bite_sequencing_history),
                self.bite_preference
                )
            
            print('=== CALLING FEEDING PLANNER ===')
            bite_sequencing_response, _ = self.gpt_interface.chat_with_openai(bite_sequencing_prompt)

            bite_sequencing_response = bite_sequencing_response.strip()
            next_bite = ast.literal_eval(bite_sequencing_response.split('Next bite as string:')[1].strip())

            if self.update_transfer_params:

                with open('prompts/' + self.transfer_parameter_prompt_file, 'r') as f:
                    transfer_params_prompt = f.read()

                transfer_params_prompt = transfer_params_prompt%(
                    str(items),
                    str(self.current_motion_params),
                    self.transfer_preference
                    )

                print('=== GETTING MOTION PARAMS ===')
                transfer_parameter_response, _ = self.gpt_interface.chat_with_openai(transfer_params_prompt)

                self.current_motion_params = ast.literal_eval(transfer_parameter_response.split('Motion parameters:')[1].strip())

                self.update_transfer_params = False

            print(f'next_bite: {next_bite}')
            for food in self.current_motion_params:
                if food[0] == next_bite:
                    next_food = food

            if self.debug:
                print(f"=== BITE SEQUENCING PROMPT ===")
                print(bite_sequencing_prompt)
                print("\n=== BITE SEQUENCING RESPONSE ===")
                print(f"BITE PREFERENCE: {self.bite_preference}\n")
                print(bite_sequencing_response)

                print(f"=== TRANSFER PARAMS PROMPT ===")
                print(transfer_params_prompt)
                print("\n=== TRANSFER PARAMS RESPONSE ===")
                print(f"TRANSFER PREFERENCE: {self.transfer_preference}\n")
                print(transfer_parameter_response)

            # TODO: Check if the values suggested are out of range. If it is modify the motion parameter list to have the max/min value.

            # # Append responses and parameters to a file
            with open(output_directory + f'motion_param_output_idx_{preference_idx}.txt', 'a') as f:
                f.write(f"=== HISTORY ===\n{history}\n")
                f.write(f"=== CURRENT MOTION PARAMS ===\n{self.current_motion_params}\n")
                if preference_change:
                    f.write(f"=== USER PREFERENCE ===\n{preference}\n")
                    f.write(f"=== BITE PREFERENCE ===\n{self.bite_preference}\n")
                    f.write(f"=== TRANSFER PREFERENCE ===\n{self.transfer_preference}\n")
                f.write(f"=== BITE SEQUENCING RESPONSE ===\n{bite_sequencing_response}\n")
                try:
                    f.write(f"=== TRANSFER PARAMS RESPONSE ===\n{transfer_parameter_response}\n")
                except:
                    pass
                f.write(f"NEXT FOOD: {next_food}\n")

            if next_bite != '':
                return next_food
            else:
                return None

        if mode == 'no_decomposer':
            with open('prompts/' + self.no_decomposer_prompt_file, 'r') as f:
                prompt = f.read()
    
            prompt = prompt%(
                str(items), 
                portions_sentence,
                str(history),
                preference
                )

            response, token_data = self.gpt_interface.chat_with_openai(prompt)
            response = response.strip()

            if self.debug:
                print(f"=== PROMPT ===")
                print(prompt)
                print(f"RESPONSE:\n{response}")

            intermediate_response = response.split('Next bite as string:')[1].strip()

            feeding_parameters = intermediate_response.split('\n')
            next_bite = ast.literal_eval(feeding_parameters[0].strip())

            if next_bite == []:
                print("NO BITES MAKE SENSE")
                return [], None, None, None, None, None
            else:
                next_bite = next_bite

            for param in feeding_parameters:
                if 'Next bite size as float:' in param:
                    bite_size = ast.literal_eval(param.split('Next bite size as float:')[1].strip())
                elif 'Next distance to mouth as float:' in param:
                    distance_to_mouth = ast.literal_eval(param.split('Next distance to mouth as float:')[1].strip())
                elif 'Next exit angle as float:' in param:
                    exit_angle = ast.literal_eval(param.split('Next exit angle as float:')[1].strip())
                elif 'Next transfer speed as float:' in param:
                    try:
                        print(param)
                        transfer_speed = ast.literal_eval(param.split('Next transfer speed as float:')[1].strip())
                    except:
                        transfer_speed = 5.0

            print(f"=== PARAMETERS ===")
            print(f"NEXT BITE: {next_bite}")
            print(f"BITE SIZE: {bite_size}")
            print(f"DISTANCE TO MOUTH: {distance_to_mouth}")
            print(f"EXIT ANGLE: {exit_angle}")
            print(f"TRANSFER SPEED: {transfer_speed}")

            # Append responses and parameters to a file
            with open(output_directory + f'prompt_v6_idx_{preference_idx}.txt', 'a') as f:
                f.write(f"=== HISTORY ===\n{history}\n")
                f.write(f"=== RESPONSE ===\n{response}\n")
                f.write(f"=== PARAMETERS ===\n")
                f.write(f"NEXT BITE: {next_bite}\n")
                f.write(f"BITE SIZE: {bite_size}\n")
                f.write(f"DISTANCE TO MOUTH: {distance_to_mouth}\n")
                f.write(f"EXIT ANGLE: {exit_angle}\n")
                f.write(f"PORTION SIZES: {portions}\n")
                f.write(f"TRANSFER SPEED: {transfer_speed}\n")
                    
            return next_bite, bite_size, distance_to_mouth, exit_angle, transfer_speed, token_data
    
    def interactive_test(self, mode):
        items = ast.literal_eval(input('Enter a list of items: '))
        portions = ast.literal_eval(input('Enter their bite portions: '))
        preference = input('What is your preference?: ')
        dips = input('What dips are available?: ')
        efficiencies = ast.literal_eval(input('Enter efficiencies: '))
        history = []
        print('---')

        while len(portions):
            efficiency_sentence = str(efficiencies)
            portions_sentence = str(portions)
            print('==== SUMMARIZED EFFICIENCIES AND PORTIONS ===')
            print(efficiency_sentence)
            print(portions_sentence)
            print('====')
            next_bites, response = self.plan(items, portions_sentence, efficiency_sentence, preference, dips, history, mode=mode)
            print('==== CHAIN-OF-THOUGHT RESPONSE ===')
            print(response)
            print('====')
            # history.append(next_bites[0])
            history += next_bites
            print("History:", history)
            for next_bite in next_bites:
                if next_bite in items:
                    idx = items.index(next_bite)
                    if portions[idx] > 1:
                        portions[idx] -= 1
                    else:
                        items.pop(idx)
                        efficiencies.pop(idx)
                        portions.pop(idx)
            print("Bites taken so far:", history)
            if not len(next_bites):
                break

    def shortlisted_runs_test(self):

        _items = ['celery', 'watermelon', 'strawberry']
        _portions = [3, 3, 3]
        _dips = ['ranch', 'chocolate sauce']
        _preferences = ['No preference', 'Celery with ranch first, then watermelon, then strawberries with chocolate', 'Strawberries with chocolate first, then watermelon, then celery with ranch']

        for preference in _preferences:

            items = _items.copy()
            portions = _portions.copy()
            dips = _dips
            efficiencies = [1.0 for _ in range(len(items))]
            history = []
            print('---')            

            while len(portions):
                efficiency_sentence = str(efficiencies)
                portions_sentence = str(portions)
                print('==== SUMMARIZED EFFICIENCIES AND PORTIONS ===')
                print(efficiency_sentence)
                print(portions_sentence)
                print('====')
                next_bites, response = self.plan(items, portions_sentence, efficiency_sentence, preference, dips, history, mode="ours")
                print('==== CHAIN-OF-THOUGHT RESPONSE ===')
                print(response)
                print('====')
                history += next_bites
                for next_bite in next_bites:
                    if next_bite in items:
                        idx = items.index(next_bite)
                        if portions[idx] > 1:
                            portions[idx] -= 1
                        else:
                            items.pop(idx)
                            efficiencies.pop(idx)
                            portions.pop(idx)
                print("Items, Bites taken so far:", items, history)
                if not len(next_bites):
                    break
    
if __name__ == '__main__':
    preference_planner = PreferencePlanner()
    preference_planner.interactive_test(mode='preference')