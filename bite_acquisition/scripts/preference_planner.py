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

#model='gpt-4-1106-preview',

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
        # [completion_tokens, prompt_tokens, total_tokens]
        tokens = [chatbot_usage.completion_tokens, chatbot_usage.prompt_tokens, chatbot_usage.total_tokens]

        print(f"=== TOKENS ===")
        print(tokens)

        return chatbot_response.strip(), tokens

class PreferencePlanner:
    def __init__(self):
        self.gpt_interface = GPTInterface()

        self.decomposer_prompt_file = 'preference_decomposer.txt'
        self.bite_sequencing_prompt_file = 'bite_sequencing.txt'
        self.transfer_parameter_prompt_file = 'transfer_parameters.txt'

    def parse_preferences(self, preference):

        with open('prompts/' + self.decomposer_prompt_file, 'r') as f:
            prompt = f.read()

        prompt = prompt%(preference)

        print(f"USER PREFERENCE: {preference}")

        response, decomposer_tokens = self.gpt_interface.chat_with_openai(prompt)
        # print(f"=== PREFERENCE_PARSER_RESPONSE ===")
        # print(response)
        intermediate_response = response.split('Bite sequence preference:')[1].strip()
        preferences = intermediate_response.split('\n')
        # print(f"INTERMEDIATE PREFERENCES: {preferences}")
        bite_preference = preferences[0]
        transfer_preference = preferences[2].split('Bite transfer preference: ')[1].strip()
        
        return bite_preference, transfer_preference, decomposer_tokens
        
    def plan(self, items, portions, efficiencies, preference, history, mode='ours'):

        dips = 0.0

        if mode == 'ours':
            print("Reading prompts from prompts/ours_new.txt")
            with open('prompts/ours.txt', 'r') as f:
                prompt = f.read()
        elif mode == 'preference':
            with open('prompts/preference_only.txt', 'r') as f:
                prompt = f.read()

        efficiency_sentence = str(efficiencies)
        portions_sentence = str(portions)

        print('==== ITEMS / PORTIONS REMAINING ===')
        print(items, portions_sentence)
        print('====')
        print('==== SUMMARIZED EFFICIENCIES ===')
        print(efficiency_sentence)
        print('==== HISTORY ===')
        print(history)

        if mode == 'ours':
            print("items:", str(items), "portions:", portions_sentence, "efficiencies:", efficiency_sentence, "preference:", preference, "dips:", str(dips), "history:", str(history))
            # check type of each argument
            if type(portions_sentence) != str or type(efficiency_sentence) != str or type(preference) != str:
                print("ERROR: type of arguments to plan() is not correct")
                exit(1)
            prompt = prompt%(str(items), portions_sentence, efficiency_sentence, preference, str(dips), str(history))
        else:
            prompt = prompt%(str(items), portions_sentence, preference, str(dips), str(history))

        response = self.gpt_interface.chat_with_openai(prompt).strip()
        print(response)
        next_bites = ast.literal_eval(response.split('Next bite as list:')[1].strip())

        return next_bites, response

    def plan_motion_primitives(self, 
                               items, 
                               portions, 
                               efficiencies, 
                               preference, 
                               bite_preference, 
                               transfer_preference, 
                               history, 
                               mode='ours'):

        dips = 0.0

        if mode == 'ours':
            print("Reading prompts from prompts/ours_new.txt")
            with open('prompts/ours.txt', 'r') as f:
                prompt = f.read()

        elif mode == 'preference':
            with open('prompts/preference_only.txt', 'r') as f:
                prompt = f.read()

        elif mode == 'motion_primitive':
            with open('prompts/old_prompt_no_dip.txt', 'r') as f:
                prompt = f.read()

        efficiency_sentence = str(efficiencies)
        portions_sentence = str(portions)
        print('==== ITEMS / PORTIONS REMAINING ===')
        print(items, portions_sentence)
        print('====')
        print('==== SUMMARIZED EFFICIENCIES ===')
        print(efficiency_sentence)
        print('==== HISTORY ===')
        print(history)

        if mode == 'ours':
            print("items:", str(items), "portions:", portions_sentence, "efficiencies:", efficiency_sentence, "preference:", preference, "dips:", str(dips), "history:", str(history))

            if type(portions_sentence) != str or type(efficiency_sentence) != str or type(preference) != str:
                print("ERROR: type of arguments to plan() is not correct")
                exit(1)
            prompt = prompt%(str(items), portions_sentence, efficiency_sentence, preference, str(dips), str(history))

        elif mode == 'motion_primitive':
            print(f"items: {items}\nportions: {portions_sentence}\nefficiencies: {efficiency_sentence}\npreference: {preference}\nbite_preference: {bite_preference}\ntransfer_preference: {transfer_preference}\ndips: {dips}\nhistory: {history}")

            if type(portions_sentence) != str or type(efficiency_sentence) != str or type(preference) != str:
                print("ERROR: type of arguments to plan() is not correct")
                exit(1)

            prompt = prompt%(str(items), portions_sentence, efficiency_sentence, preference, bite_preference, transfer_preference, str(history))

        else:
            prompt = prompt%(str(items), portions_sentence, preference, str(dips), str(history))

        # print("Prompt:\n", prompt)

        response = self.gpt_interface.chat_with_openai(prompt)[0].strip()
        print(f" RESPONSE:\n{response}")
        # intermediate_response = response.split('Next bite as list:')[1]
        # next_bites = ast.literal_eval(intermediate_response.split('Next bite size:')[0].strip())
        # print(f"next_bites: {(next_bites)}")
        # bite_size = ast.literal_eval(intermediate_response.split('Next bite size as float:')[1].strip())
        # print(f"bite_size: {(bite_size)}")
        intermediate_response = response.split('Next bite as list:')[1].strip()
        feeding_parameters = intermediate_response.split('\n')
        next_bites = ast.literal_eval(feeding_parameters[0])
        bite_size = ast.literal_eval(feeding_parameters[2].split('Next bite size as float:')[1].strip())
        distance_to_mouth = ast.literal_eval(feeding_parameters[4].split('Next distance to mouth as float:')[1].strip())
        exit_angle = ast.literal_eval(feeding_parameters[6].split('Next exit angle as float:')[1].strip())
        print(f"next_bites: {next_bites} (type: {type(next_bites)})")
        print(f"bite_size: {bite_size} (type: {type(bite_size)})")
        print(f"distance_to_mouth: {distance_to_mouth} (type: {type(distance_to_mouth)})")
        print(f"exit_angle: {exit_angle} (type: {type(exit_angle)})")
        
        # print(f"next_bites: {(next_bites)}\nbite_size: {(bite_size)}")
        
        return next_bites, bite_size, distance_to_mouth, exit_angle
    
    def plan_decomposer(self, items, portions, efficiencies, preference, history):

        # Extracting bite preference and transfer preference
        bite_preference, transfer_preference, decomposer_tokens = self.parse_preferences(preference)

        # Extracting bite sequencing history and transfer parameters history
        bite_sequencing_history = [item[:2] for item in history]
        transfer_param_history = [[item[0]] + item[2:] for item in history]

        # Reading prompts
        with open('prompts/' + self.bite_sequencing_prompt_file, 'r') as f:
            bite_sequencing_prompt = f.read()

        with open('prompts/' + self.transfer_parameter_prompt_file, 'r') as f:
            transfer_params_prompt = f.read()

        portions_sentence = str(portions)
        efficiency_sentence = str(efficiencies)
        print('\n==== ITEMS / PORTIONS REMAINING ===')
        print(f"{items} / {portions_sentence}")
        print('==== PREFERENCES ===')
        print(f"USER PREFERENCE: {preference}")
        print(f"BITE PREFERENCE: {bite_preference}")
        print(f"TRANSFER PREFERENCE: {transfer_preference}")
        print('==== HISTORY ===')
        print(history)
        
        bite_sequencing_prompt = bite_sequencing_prompt%(str(items), portions_sentence, efficiency_sentence, bite_preference, str(bite_sequencing_history))

        bite_sequencing_response, bite_sequencing_tokens = self.gpt_interface.chat_with_openai(bite_sequencing_prompt)
        bite_sequencing_response = bite_sequencing_response.strip()
        print("\n=== BITE SEQUENCING RESPONSE ===")
        print(f"BITE PREFERENCE:, {bite_preference}\n")
        print(bite_sequencing_response)
        bite_response = bite_sequencing_response.split('Next bite as list:')[1].strip()
        bite_parameters = bite_response.split('\n')
        next_bite = ast.literal_eval(bite_parameters[0])
        bite_size = ast.literal_eval(bite_parameters[2].split('Next bite size as float:')[1].strip())

        transfer_params_prompt = transfer_params_prompt%(transfer_preference, next_bite[0], str(transfer_param_history))
        transfer_parameter_response, transfer_param_tokens = self.gpt_interface.chat_with_openai(transfer_params_prompt)
        transfer_parameter_response = transfer_parameter_response.strip()
        print("\n=== TRANSFER PARAMS RESPONSE ===")
        print(f"TRANSFER PREFERENCE: {transfer_preference}\n")
        print(transfer_parameter_response)
        transfer_response = transfer_parameter_response.split('Next distance to mouth as float:')[1].strip()
        transfer_parameters = transfer_response.split('\n')
        distance_to_mouth = ast.literal_eval(transfer_parameters[0])
        exit_angle = ast.literal_eval(transfer_parameters[2].split('Next exit angle as float:')[1].strip())

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

        print("\n=== TOKENS ===")
        print(token_usage)

        return next_bite, bite_size, distance_to_mouth, exit_angle, token_usage

    def plan_no_decomposer(self, items, portions, efficiencies, preference, history, mode='ours'):

        if mode == 'ours':
            print("Reading prompts from prompts/ours_new.txt")
            with open('prompts/ours.txt', 'r') as f:
                prompt = f.read()

        elif mode == 'preference':
            with open('prompts/preference_only.txt', 'r') as f:
                prompt = f.read()

        elif mode == 'motion_primitive':
            with open('prompts/prompt_no_decomposition.txt', 'r') as f:
                prompt = f.read()

        efficiency_sentence = str(efficiencies)
        portions_sentence = str(portions)
        print('==== ITEMS / PORTIONS REMAINING ===')
        print(items, portions_sentence)
        print('==== SUMMARIZED EFFICIENCIES ===')
        print(efficiency_sentence)
        print('==== PREFERENCES ===')
        print(f"user preference: {preference}")
        print('==== HISTORY ===')
        print(history)

        if mode == 'ours':
            print("items:", str(items), "portions:", portions_sentence, "efficiencies:", efficiency_sentence, "preference:", preference, "dips:", str(dips), "history:", str(history))

            if type(portions_sentence) != str or type(efficiency_sentence) != str or type(preference) != str:
                print("ERROR: type of arguments to plan() is not correct")
                exit(1)
            prompt = prompt%(str(items), portions_sentence, efficiency_sentence, preference, str(dips), str(history))

        elif mode == 'motion_primitive':
            if type(portions_sentence) != str or type(efficiency_sentence) != str or type(preference) != str:
                print("ERROR: type of arguments to plan() is not correct")
                exit(1)

            prompt = prompt%(str(items), portions_sentence, efficiency_sentence, preference, str(history))

        else:
            prompt = prompt%(str(items), portions_sentence, preference, str(history))

        response, token_data = self.gpt_interface.chat_with_openai(prompt)
        response = response.strip()
        print(f"RESPONSE:\n{response}")
        intermediate_response = response.split('Next bite as list:')[1].strip()
        feeding_parameters = intermediate_response.split('\n')
        next_bites = ast.literal_eval(feeding_parameters[0])
        bite_size = ast.literal_eval(feeding_parameters[2].split('Next bite size as float:')[1].strip())
        distance_to_mouth = ast.literal_eval(feeding_parameters[4].split('Next distance to mouth as float:')[1].strip())
        exit_angle = ast.literal_eval(feeding_parameters[6].split('Next exit angle as float:')[1].strip())
                
        return next_bites, bite_size, distance_to_mouth, exit_angle, token_data

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
    # preference_planner.interactive_test(mode='preference')
    # preference_planner.shortlisted_runs_test()
    # EXAMPLE INPUTS
    # Enter a list of items: ['banana', 'brownie', 'chocolate sauce']
    # Enter their bite portions: [1, 2]  
    # Enter their efficiencies: [1, 1]
    # What is your preference?: Dip brownie bites in chocolate dip
