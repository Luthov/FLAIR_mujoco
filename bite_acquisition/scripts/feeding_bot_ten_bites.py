# from speech_to_text.speech_to_text import get_user_preference
from preference_planner import PreferencePlanner

from preferences import ethan_interview_preferences, matthew_interview_preferences, hau_wen_interview_preferences, jonathan_interview_preferences, yi_heng_interview_preferences, amirul_interview_preferences, janssen_interview_preferences, ben_interview_preferences, aaradh_interview_preferences, darren_interview_preferences, luke_preferences, interrupt_preferences, interview_food_items # , modified_interview_preferences

class FeedingBot:
    def __init__(self):

        self.preference_planner = PreferencePlanner()

        print("Feeding Bot initialized\n")

        self.execute = False
        self.preference_interrupt = True
        self.speech_to_text = False
        self.modified = False

        self.bite_portion = 1.0

        # self.participant_list = ['ethan', 'matthew', 'hauwen', 'jonathan']
        # self.participant_list = ['matthew', 'ethan', 'hauwen', 'jonathan']
        # self.participant_list = ['ben', 'aaradh', 'darren']
        self.participant_list = ['luke']

        # Choose to use decomposer or not
        self.mode = 'decomposer'
        # self.mode = 'decomposer'
        self.decomposer_output_directory = 'feeding_bot_output/interview_outputs/prompt_improvements/decomposer_outputs/'
        self.no_decomposer_output_directory = 'feeding_bot_output/interview_outputs/prompt_improvements/'

        if self.mode == 'decomposer':
            self.output_directory = self.decomposer_output_directory
            print('=== USING DECOMPOSER PROMPT ===')
        elif self.mode == 'no_decomposer':
            self.output_directory = self.no_decomposer_output_directory
            print('=== USING NON DECOMPOSER PROMPT ===')

        # self.preferences = range(len(ethan_interview_preferences))
        self.preferences = [0]

    def clear_plate(self):

        for participant in self.participant_list:

            if participant == 'ethan':
                interview_preferences = ethan_interview_preferences
            elif participant == 'matthew':
                interview_preferences = matthew_interview_preferences
            elif participant == 'hauwen':
                interview_preferences = hau_wen_interview_preferences
            elif participant == 'jonathan':
                interview_preferences = jonathan_interview_preferences
            elif participant == 'yiheng':
                interview_preferences = yi_heng_interview_preferences
            elif participant == 'amirul':
                interview_preferences = amirul_interview_preferences
            elif participant == 'janssen':
                interview_preferences = janssen_interview_preferences
            elif participant == 'ben':
                interview_preferences = ben_interview_preferences
            elif participant == 'aaradh':
                interview_preferences = aaradh_interview_preferences
            elif participant == 'darren':
                interview_preferences = darren_interview_preferences

            interview_preferences = luke_preferences

            self.output_directory = f'feeding_bot_output/{participant}/10_bite_testing/'
        
            for preference_idx in self.preferences: # range(len(icorr_preferences)):

                if self.speech_to_text:
                    user_preference = get_user_preference()
                else:
                    user_preference = interview_preferences[preference_idx]
                    if self.modified:
                        user_preference = modified_interview_preferences[preference_idx]
                    if user_preference == "":
                        continue

                self.items = [interview_food_items[preference_idx]]
                food_items = self.items[0]
                
                if len(food_items) == 3:
                    self.item_portions = [3.0] * len(food_items)
                    actions_remaining = 9
                    # action_interrupt = (actions_remaining < 5)
                else:
                    self.item_portions = [3.0] * len(food_items)
                    actions_remaining = 12

                # Bite history
                bite_history = []
                # Token history
                token_history = []

                preference_change = True
                not_changed = True
                start = True

                sequence_idx = 0
                
                while actions_remaining:

                    print(f"=== RUN NUMBER ===")
                    print(preference_idx)
                    print(f"=== ACTIONS REMAINING ===")
                    print(actions_remaining)
                    print(f"=== USER PREFERENCE ===")
                    print(user_preference)

                    action_interrupt = (actions_remaining < 6)

                    if self.preference_interrupt & action_interrupt & not_changed:
                        # Get user preferences
                        print("=== CURRENT USER PREFERENCE ===")
                        print(user_preference)
                        # new_user_preference = input("Do you want to update your preference? Otherwise input [n] or Enter to continue\n")
                        user_preference = interrupt_preferences[preference_idx] 
                        # if new_user_preference not in ['n', '']:
                        #     user_preference = new_user_preference
                        print("=== NEW USER PREFERENCE ===")
                        print(user_preference)

                        preference_change = True
                        not_changed = False
                        with open(self.output_directory + f'motion_param_output_idx_{preference_idx}.txt', 'a') as f:
                            f.write(f"=== NEW USER PREFERENCE ===\n{user_preference}\n")

                    food_portion_rounded = [round(portion) for portion in self.item_portions]

                    if start or preference_change:
                        feeding_sequence = self.preference_planner.plan(
                            food_items, 
                            food_portion_rounded, 
                            user_preference, 
                            bite_history,
                            preference_change,
                            preference_idx,
                            self.mode,
                            self.output_directory
                        )

                        print('=== FEEDING SEQUENCE ===')
                        print(feeding_sequence)
                        start = False
                        preference_change = False
                    
                    next_bite = feeding_sequence[sequence_idx]
                    next_food_item = next_bite[0]

                    actions_remaining -= 1
                        
                    for idx in range(len(food_items)):
                        if food_items[idx] == next_food_item:
                            self.item_portions[idx] -= self.bite_portion
                            self.item_portions[idx] = round(self.item_portions[idx], 2)
                            break
                    
                    input('BITE SUCCESSFUL?')
                    bite_history.append(next_bite)
                    sequence_idx += 1

                    print('=== HISTORY ===')
                    print(bite_history)

                    if actions_remaining == 0 or (actions_remaining == len(feeding_sequence)):
                        with open(self.output_directory + f'histories_idx_{preference_idx}.txt', 'a') as f:
                            f.write(f"=== FINAL HISTORY ===\n{bite_history}\n")
                            # f.write(f"=== FINAL TOKEN HISTORY ===\n{token_history}\n")
                            f.write(f"=== USER PREFERENCE ===\n{user_preference}\n")
                            break

if __name__ == "__main__":
    feeding_bot = FeedingBot()
    feeding_bot.clear_plate()
