import cv2
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
        return chatbot_response.strip()

class PreferencePlanner:
    def __init__(self):
        self.gpt_interface = GPTInterface()

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

    def plan_motion_primitives(self, items, portions, efficiencies, preference, bite_preference, transfer_preference, bite_size, history, mode='ours'):

        dips = 0.0

        if mode == 'ours':
            print("Reading prompts from prompts/ours_new.txt")
            with open('prompts/ours.txt', 'r') as f:
                prompt = f.read()

        elif mode == 'preference':
            with open('prompts/preference_only.txt', 'r') as f:
                prompt = f.read()

        elif mode == 'motion_primitive':
            # with open('prompts/motion_primitive_v3.txt', 'r') as f:
            with open('prompts/motion_primitive_v4_transfer_params.txt', 'r') as f:
                prompt = f.read()

        efficiency_sentence = str(efficiencies)
        portions_sentence = str(portions)
        print('==== ITEMS / PORTIONS REMAINING ===')
        print(items, portions_sentence)
        print('====')
        print('==== SUMMARIZED EFFICIENCIES ===')
        print(efficiency_sentence)
        print('==== BITE SIZE ===')
        print(bite_size)
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

            prompt = prompt%(str(items), portions_sentence, efficiency_sentence, preference, bite_preference, transfer_preference, str(dips), str(history))

        else:
            prompt = prompt%(str(items), portions_sentence, preference, str(dips), str(history))

        # print("Prompt:\n", prompt)

        response = self.gpt_interface.chat_with_openai(prompt).strip()
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
        entry_angle = ast.literal_eval(feeding_parameters[6].split('Next entry angle as float:')[1].strip())
        exit_angle = ast.literal_eval(feeding_parameters[8].split('Next exit angle as float:')[1].strip())
        print(f"next_bites: {next_bites} (type: {type(next_bites)})")
        print(f"bite_size: {bite_size} (type: {type(bite_size)})")
        print(f"distance_to_mouth: {distance_to_mouth} (type: {type(distance_to_mouth)})")
        print(f"entry_angle: {entry_angle} (type: {type(entry_angle)})")
        print(f"exit_angle: {exit_angle} (type: {type(exit_angle)})")
        
        # print(f"next_bites: {(next_bites)}\nbite_size: {(bite_size)}")
        
        return next_bites, bite_size, response

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
