import ast

for preference_idx in range(10):
    history_file = f'interview_outputs/interview_aaradh/histories_idx_{preference_idx}.txt'
    try:
        with open(history_file, 'r') as file:
            test = file.read()
            bite_sequence = test.split("\n")[1]
            # bite_sequence = "[['salmon', 5.0, 7.5, 90.0, 5.0], ['broccoli', 5.0, 7.5, 90.0, 5.0], ['salmon', 5.0, 7.5, 90.0, 5.0], ['salmon', 5.0, 7.5, 90.0, 5.0], ['broccoli', 5.0, 7.5, 90.0, 5.0], ['broccoli', 5.0, 7.5, 90.0, 5.0], ['mashed potatoes', 5.0, 7.5, 90.0, 5.0], ['mashed potatoes', 5.0, 7.5, 90.0, 5.0], ['mashed potatoes', 5.0, 7.5, 90.0, 5.0]]"
            parsed_test = ast.literal_eval(bite_sequence)
            first_elements = [item[0] for item in parsed_test]
            print(f'=== PLATE {preference_idx +1} ===')
            print(first_elements)
    except FileNotFoundError:
        continue