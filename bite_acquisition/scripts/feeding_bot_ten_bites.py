# from speech_to_text.speech_to_text import get_user_preference
from preference_planner import PreferencePlanner

from preferences import ethan_interview_preferences, matthew_interview_preferences, hau_wen_interview_preferences, jonathan_interview_preferences, yi_heng_interview_preferences, amirul_interview_preferences, janssen_interview_preferences, ben_interview_preferences, aaradh_interview_preferences, darren_interview_preferences, luke_preferences, interrupt_preferences, interview_food_items # , modified_interview_preferences

class FeedingBot:
    def __init__(self):

        self.preference_planner = PreferencePlanner()

        print("Feeding Bot initialized\n")

        self.execute = False
        self.preference_interrupt = False
        self.speech_to_text = False
        self.modified = False

        self.bite_portion = 1.0

        # Choose to use decomposer or not
        self.mode = 'decomposer'
        # self.mode = 'decomposer'
        self.decomposer_output_directory = 'feeding_bot_output/user_study/hauwen/'
        self.no_decomposer_output_directory = 'feeding_bot_output/interview_outputs/prompt_improvements/'

        if self.mode == 'decomposer':
            self.output_directory = self.decomposer_output_directory
            print('=== USING DECOMPOSER PROMPT ===')
        elif self.mode == 'no_decomposer':
            self.output_directory = self.no_decomposer_output_directory
            print('=== USING NON DECOMPOSER PROMPT ===')

        # self.preferences = range(4)
        self.preferences = [1]

    def clear_plate(self):

        self.items = [
                ["rice", "chicken", "mixed vegetables"],
                ["rice", "chicken", "cucumber"],
                ["rice", "broccoli", "beef", "orange"],
                ["mashed potatoes", "steak", "green beans", "carrots"]
            ]
        
        yi_heng_interview_preferences = [
            "I want to alternate between the foods in this order: chicken, then rice, then mixed vegetables",
            "I want alternating bites of food in this order: chicken, rice and then cucumber.",
            "I want alternating bites of beef, then broccoli then rice. Finally, feed me the oranges.",
            "I want to have alternating bites of food in this order: steak, then green beans, then carrots and then mashed potatoes"
        ]

        chris_interview_preferences = [
            "I prefer to eat vegetables with rice and leave the chicken for the last.",
            "Rice with chicken and lastly, cucumber",
            "Different random sequences",
            # "No preference",
            "I prefer to finish my mashed potatoes or carrots or green beans first and finish my steak last but with a mixture of different food sequences."
            # "Mashed potatoes, carrots, green beans, steak"
        ]
        benjamin_interview_preferences = [
            "I would like to start with chicken, then alternate between all the other foods. ",
            "I would like to finish all the cucumbers first, then alternate between the other foods. ",
            "First, alternate between all the foods except the oranges. I would like to have the oranges as dessert. ",
            "I would like to finish all the mash potatoes, green beans and carrots first in an alternating pattern. Then all the steak last. "
        ]
        vassanth_interview_preferences = [
            "I have no preference.",
            # "Feed me mostly rice followed by chicken and cucumber. ",
            "Feed me mostly rice followed by alternating between chicken and cucumber. ",
            "Feed me broccoli first and once that finishes alternate between beef, orange and rice.",
            "Feed me all the steak first then all the mashed potatoes then alternate between carrots and green beans"
        ]
        
        ritesh_interview_preferences = [
            "veggies first then the rest. Try to alternate textures for the 'rest'.",
            "cucumbers first, then an alternating combination of rice and chicken",
            "i can't eat beef, the rest just randomise the textures.",
            "alternate all the textures, no other preference of order."
        ]

        gabriel_interview_preferences = [
            "Please alternate between the food",
            "I want to eat chicken, rice and cucumber in sequence",
            "I want to finish the broccoli first, then alternate between the other items",
            "I want to finish the steak first and then alternate between items"
        ]

        interview_preferences = [
            "Alternate between rice and other foods. Leave more chicken at the end",
            "Alternate between rice and other foods, leave more chicken at the end",
            "Alternate between rice and other foods, leave all the oranges to the end",
            "Alternate between mashed potatoes and other foods, leave more steak at the end"
        ]

        interview_preferences = [
            "Alternate between rice and other foods. Leave more chicken at the end. For other foods also alternate between chicken and vegetables.",
            "Alternate between rice and other foods, leave more chicken at the end. for other foods alternate between chicken and cucumber",
            "",
            "Alternate between mashed potatoes and other foods, leave more steak at the end. Dont have 3 of the same foods consecutively"
        ]
    
        for preference_idx in self.preferences: # range(len(icorr_preferences)):

            if self.speech_to_text:
                user_preference = get_user_preference()

            else:
                user_preference = interview_preferences[preference_idx]
                if self.modified:
                    user_preference = modified_interview_preferences[preference_idx]
                if user_preference == "":
                    continue

            # self.items = [interview_food_items[preference_idx]]
            food_items = self.items[preference_idx]
            
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
            
            # while actions_remaining:

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
                user_preference = interrupt_preferences[preference_idx] 
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

                with open(self.output_directory + f'feeding_idx_{preference_idx}.txt', 'a') as f:
                    f.write(f"=== FEEDING SEQUENCE ===\n{feeding_sequence}\n")
                    # f.write(f"=== FINAL TOKEN HISTORY ===\n{token_history}\n")
                    f.write(f"=== USER PREFERENCE ===\n{user_preference}\n")

                #     start = False
                #     preference_change = False
                
                # next_bite = feeding_sequence[sequence_idx]
                # next_food_item = next_bite[0]

                # actions_remaining -= 1
                    
                # for idx in range(len(food_items)):
                #     if food_items[idx] == next_food_item:
                #         self.item_portions[idx] -= self.bite_portion
                #         self.item_portions[idx] = round(self.item_portions[idx], 2)
                #         break
                
                # input('BITE SUCCESSFUL?')
                # bite_history.append(next_bite)
                # sequence_idx += 1

                # print('=== HISTORY ===')
                # print(bite_history)

                # if actions_remaining == 0 or (actions_remaining == len(feeding_sequence)):
                #     with open(self.output_directory + f'histories_idx_{preference_idx}.txt', 'a') as f:
                #         f.write(f"=== FINAL HISTORY ===\n{bite_history}\n")
                #         # f.write(f"=== FINAL TOKEN HISTORY ===\n{token_history}\n")
                #         f.write(f"=== USER PREFERENCE ===\n{user_preference}\n")
                #         break


if __name__ == "__main__":
    feeding_bot = FeedingBot()
    feeding_bot.clear_plate()
