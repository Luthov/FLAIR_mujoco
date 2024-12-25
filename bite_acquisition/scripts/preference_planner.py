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
                   model='gpt-4-0125-preview',
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

        self.decomposer_prompt_file = 'ICORR_prompts_v4/decomposer_prompts/decomposer.txt'
        self.bite_sequencing_prompt_file = 'ICORR_prompts_v4/decomposer_prompts/bite_acquisition_flair.txt'
        self.transfer_parameter_prompt_file = 'ICORR_prompts_v4/decomposer_prompts/bite_transfer.txt'

        self.no_decomposer_prompt_file = 'flair_testing/flair_v8.txt'
        
        self.flair_prompt = True
        self.debug = False

    def parse_preferences(self, preference):

        with open('prompts/' + self.decomposer_prompt_file, 'r') as f:
            prompt = f.read()

        prompt = prompt%(preference)

        response, decomposer_tokens = self.gpt_interface.chat_with_openai(prompt)

        intermediate_response = response.split('Bite acquisition preference:')[1].strip()
        preferences = intermediate_response.split('\n')
        bite_preference = preferences[0]
        for pref in preferences:
            if 'Bite transfer preference:' in pref:
                transfer_preference = pref.split('Bite transfer preference: ')[1].strip()

        print(f"=== USER PREFERENCE ===")
        print(preference)
        print(f"=== BITE PREFERENCE ===")
        print(bite_preference)
        print(f"=== TRANSFER PREFERENCE ===")
        print(transfer_preference)
        
        return bite_preference, transfer_preference, decomposer_tokens
    
    def plan(self, 
             items, 
             portions, 
             efficiencies, 
             preference, 
             history, 
             preference_idx, 
             mode,
             output_directory):

        portions_sentence = str(portions)
        efficiency_sentence = str(efficiencies)
        print('\n==== ITEMS / PORTIONS REMAINING ===')
        print(f"{items} / {portions_sentence}")
        print('==== HISTORY ===')
        print(history)
    
        if mode == 'decomposer':

            # Extracting bite preference and transfer preference
            bite_preference, transfer_preference, decomposer_tokens = self.parse_preferences(preference)

            # Reading prompts
            with open('prompts/' + self.bite_sequencing_prompt_file, 'r') as f:
                bite_sequencing_prompt = f.read()

            with open('prompts/' + self.transfer_parameter_prompt_file, 'r') as f:
                transfer_params_prompt = f.read()

            # Extracting bite sequencing history and transfer parameters history
            bite_sequencing_history = [item[:2] for item in history]
            transfer_param_history = [[item[0]] + item[2:] for item in history]

            bite_sequencing_prompt = bite_sequencing_prompt%(
                bite_preference, 
                str(items), 
                portions_sentence, 
                efficiency_sentence, 
                str(bite_sequencing_history)
                )

            bite_sequencing_response, bite_sequencing_tokens = self.gpt_interface.chat_with_openai(bite_sequencing_prompt)

            bite_sequencing_response = bite_sequencing_response.strip()
            bite_response = bite_sequencing_response.split('Next food item as string:')[1].strip()
            bite_parameters = bite_response.split('\n')
            next_bite = ast.literal_eval(bite_parameters[0])
            for param in bite_parameters:
                if 'Next bite size as float:' in param:
                    bite_size = ast.literal_eval(param.split('Next bite size as float:')[1].strip())

            transfer_params_prompt = transfer_params_prompt%(
                transfer_preference, 
                next_bite, 
                str(transfer_param_history)
                )

            transfer_parameter_response, transfer_param_tokens = self.gpt_interface.chat_with_openai(transfer_params_prompt)

            transfer_parameter_response = transfer_parameter_response.strip()
            transfer_response = transfer_parameter_response.split('Next distance to mouth as float:')[1].strip()
            transfer_parameters = transfer_response.split('\n')
            distance_to_mouth = ast.literal_eval(transfer_parameters[0])
            for param in transfer_parameters:
                if 'Next exit angle as float:' in param:
                    exit_angle = ast.literal_eval(param.split('Next exit angle as float:')[1].strip())

            if self.debug:
                print(f"=== BITE SEQUENCING PROMPT ===")
                print(bite_sequencing_prompt)
                print("\n=== BITE SEQUENCING RESPONSE ===")
                print(f"BITE PREFERENCE: {bite_preference}\n")
                print(bite_sequencing_response)

                print(f"=== TRANSFER PARAMS PROMPT ===")
                print(transfer_params_prompt)
                print("\n=== TRANSFER PARAMS RESPONSE ===")
                print(f"TRANSFER PREFERENCE: {transfer_preference}\n")
                print(transfer_parameter_response)

            print("\n=== PARAMETERS ===")
            print("NEXT BITE:", next_bite)
            print("BITE SIZE:", bite_size)
            print("DISTANCE TO MOUTH:", distance_to_mouth)
            print("EXIT ANGLE:", exit_angle)

            # Arranging tokens
            token_usage = {}
            token_usage['decomposer_tokens'] = decomposer_tokens
            token_usage['bite_sequencing_tokens'] = bite_sequencing_tokens
            token_usage['transfer_param_tokens'] = transfer_param_tokens

            # Append responses and parameters to a file
            with open(output_directory + f'decomposer_output_idx_{preference_idx}.txt', 'a') as f:
                f.write(f"=== HISTORY ===\n{history}\n")
                f.write(f"=== BITE PREFERENCE ===\n{bite_preference}\n")
                f.write(f"=== BITE SEQUENCING RESPONSE ===\n{bite_sequencing_response}\n")
                f.write(f"=== TRANSFER PREFERENCE ===\n{transfer_preference}\n")
                f.write(f"=== TRANSFER PARAMS RESPONSE ===\n{transfer_parameter_response}\n")
                f.write(f"=== PARAMETERS ===\n")
                f.write(f"NEXT BITE: {next_bite}\n")
                f.write(f"BITE SIZE: {bite_size}\n")
                f.write(f"DISTANCE TO MOUTH: {distance_to_mouth}\n")
                f.write(f"EXIT ANGLE: {exit_angle}\n")
                f.write(f"PORTION SIZES: {portions}\n")

            return next_bite, bite_size, distance_to_mouth, exit_angle, token_usage   

        if mode == 'no_decomposer':
            with open('prompts/' + self.no_decomposer_prompt_file, 'r') as f:
                prompt = f.read()
    
            prompt = prompt%(
                str(items), 
                portions_sentence, 
                preference, 
                str(history)
                )

            response, token_data = self.gpt_interface.chat_with_openai(prompt)
            response = response.strip()

            if self.debug:
                print(f"=== PROMPT ===")
                print(prompt)
                print(f"RESPONSE:\n{response}")

            if self.flair_prompt:
                intermediate_response = response.split('Next bite as list:')[1].strip()
            else:
                intermediate_response = response.split('Next food item as string:')[1].strip()

            feeding_parameters = intermediate_response.split('\n')
            next_bite = ast.literal_eval(feeding_parameters[0].strip())

            if next_bite == []:
                print("NO BITES MAKE SENSE")
                return [], None, None, None, None
            else:
                next_bite = next_bite[0]

            for param in feeding_parameters:
                if 'Next bite size as float:' in param:
                    bite_size = ast.literal_eval(param.split('Next bite size as float:')[1].strip())
                elif 'Next distance to mouth as float:' in param:
                    distance_to_mouth = ast.literal_eval(param.split('Next distance to mouth as float:')[1].strip())
                elif 'Next exit angle as float:' in param:
                    exit_angle = ast.literal_eval(param.split('Next exit angle as float:')[1].strip())

            print(f"=== PARAMETERS ===")
            print(f"NEXT BITE: {next_bite}")
            print(f"BITE SIZE: {bite_size}")
            print(f"DISTANCE TO MOUTH: {distance_to_mouth}")
            print(f"EXIT ANGLE: {exit_angle}")

            # Append responses and parameters to a file
            with open(output_directory + f'flair_output_idx_{preference_idx}.txt', 'a') as f:
                f.write(f"=== HISTORY ===\n{history}\n")
                f.write(f"=== RESPONSE ===\n{response}\n")
                f.write(f"=== PARAMETERS ===\n")
                f.write(f"NEXT BITE: {next_bite}\n")
                f.write(f"BITE SIZE: {bite_size}\n")
                f.write(f"DISTANCE TO MOUTH: {distance_to_mouth}\n")
                f.write(f"EXIT ANGLE: {exit_angle}\n")
                f.write(f"PORTION SIZES: {portions}\n")
                    
            return next_bite, bite_size, distance_to_mouth, exit_angle, token_data
    
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