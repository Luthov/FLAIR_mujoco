import ast
import csv
import os

csv_file_path = 'all_histories.csv'

# Initialize the CSV file with headers if it doesn't exist
if not os.path.exists(csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        headers = []
        for preference_idx in range(20):
            headers.append(f'Preference Index {preference_idx} - No Decomposer')
            headers.append(f'Preference Index {preference_idx} - Decomposer')
        csvwriter.writerow(headers)

data_to_append = []

# Collect data for each preference index
for preference_idx in range(20):
    no_decomposer_file_path = f'no_decomposer_output/histories_idx_{preference_idx}.txt'
    decomposer_file_path = f'decomposer_output/histories_idx_{preference_idx}.txt'

    with open(no_decomposer_file_path, 'r') as f:
        output = f.read()
        no_decomposer_history = output.split('\n')[1]
        no_decomposer_history_list = ast.literal_eval(no_decomposer_history)
        
    with open(decomposer_file_path, 'r') as f:
        output = f.read()
        decomposer_history = output.split('\n')[1]
        decomposer_history_list = ast.literal_eval(decomposer_history)
        
    max_len = max(len(no_decomposer_history_list), len(decomposer_history_list))

    for i in range(max_len):
        if len(data_to_append) <= i:
            data_to_append.append([''] * 40)  # 20 preference indices * 2 columns each
        if i < len(no_decomposer_history_list):
            data_to_append[i][preference_idx * 2] = no_decomposer_history_list[i]
        if i < len(decomposer_history_list):
            data_to_append[i][preference_idx * 2 + 1] = decomposer_history_list[i]

# Write the collected data to the CSV file
with open(csv_file_path, 'a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(data_to_append)