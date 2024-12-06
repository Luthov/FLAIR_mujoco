import ast
import csv
import os

icorr_preferences = [
    "Feed me all the rice first, then alternate between chicken and vegetables",
    "I want alternate bites of chicken and rice. I prefer to be fed larger bites and for the spoon to be further away from me.",
    "I only want meat. Tilt the spoon slightly when feeding me. Feed me with larger bites", # Rerun idx 2
    "Start with the vegetables, then the meat. Keep the bites small.",
    "Alternate between rice and vegetables, but do not feed me chicken. Keep the spoon far from my mouth.", # Rerun idx 4
    "Feed me all the chicken first, then the rice. Use smaller bites and be careful not to tilt the spoon too much.",
    "I prefer alternate bites of rice, chicken and vegetables. Do not repeat any bites. Feed me evenly without tilting the spoon.",
    "Give me two bites of meat first, then alternate between vegetables and rice. Make sure to tilt the spoon a little higher", # Rerun idx 7
    "I want all the vegetables first, followed by alternating bites of chicken and rice. Keep the bites medium-sized and keep the spoon far from me.",
    "Avoid the vegetables and give me only rice and chicken. Keep the bites small and tilt the spoon slightly.",
    "Start with the chicken, then the vegetables, and end with the rice. Keep the spoon close to me.",
    "Feed me only the rice, one spoonful at a time. Keep the bites large.",
    "Alternate between rice, chicken, and vegetables. Keep the spoon at a distance and use small bites.",
    "I want one bite of chicken followed by two bites of rice. Feed me with tilted spoonfuls.", # Rerun idx 13
    "Feed me vegetables first, then alternate rice and meat. Use small bites and tilt the spoon slightly. Also keep the spoon close to me.",
    "Give me larger bites of chicken, followed by smaller bites of rice. Feed me with a tilt in the spoon and do not come too close to me.", # Rerun idx 15
    "Feed me rice first, then alternate between chicken and vegetables. Keep the spoon tilted slightly upwards.",
    "I only want vegetables. Feed me in small bites..",
    "Start with a bite of vegetables, then alternate between chicken and rice.",
    "I have no preference in the sequence, but I prefer the spoon to be closer to me."
]

csv_file_path = 'flair_vs_ours_last_one_pls.csv'

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
    no_decomposer_file_path = f'icorr_outputs_v4/flair_output/histories_idx_{preference_idx}.txt'
    decomposer_file_path = f'icorr_outputs_v4/decomposer_output/histories_idx_{preference_idx}.txt'

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

    # Append preferences to the data_to_append list for no_decomposer column
    if len(data_to_append) <= max_len:
        data_to_append.append([''] * 40)
    data_to_append[max_len][preference_idx * 2] = icorr_preferences[preference_idx]

    # Every 4 preference indices, append the next set of data underneath the existing one
    if (preference_idx + 1) % 4 == 0:
        max_len += 1

# Write the collected data to the CSV file
with open(csv_file_path, 'a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(data_to_append)