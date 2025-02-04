import ast

for preference_idx in [0,1,2,4,6]:
    history_file = f'interview_outputs/interview_amirul/modified/histories_idx_{preference_idx}.txt'
    try:
        with open(history_file, 'r') as file:
            test = file.read()
            bite_sequence = test.split("\n")[1]
            parsed_test = ast.literal_eval(bite_sequence)
            first_elements = [item[0] for item in parsed_test]
            print(f'=== PLATE {preference_idx +1} ===')
            print(first_elements)
    except FileNotFoundError:
        continue