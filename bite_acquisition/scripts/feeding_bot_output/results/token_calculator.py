import ast 
import csv

decomposer_total_sum_run = []
no_decomposer_total_sum_run = []

for preference_idx in range(20): 
    no_decomposer_file_path = f'icorr_outputs_v4/flair_output/histories_idx_{preference_idx}.txt'
    decomposer_file_path = f'icorr_outputs_v4/decomposer_output/histories_idx_{preference_idx}.txt'

    with open(no_decomposer_file_path, 'r') as f:
        output = f.read()
        no_decomposer_token = output.split('\n')[3]
        no_decomposer_token_list = ast.literal_eval(no_decomposer_token)
        
    with open(decomposer_file_path, 'r') as f:
        output = f.read()
        decomposer_token = output.split('\n')[3]
        decomposer_token_list = ast.literal_eval(decomposer_token)

    decomposer_sum = [d['decomposer_tokens'][-1] + d['bite_sequencing_tokens'][-1] + d['transfer_param_tokens'][-1] for d in decomposer_token_list]

    decomposer_total_sum = sum(decomposer_sum)
    no_decomposer_sum = sum(sublist[-1] for sublist in no_decomposer_token_list)

    decomposer_total_sum_run.append(decomposer_total_sum)
    no_decomposer_total_sum_run.append(no_decomposer_sum)

print(decomposer_total_sum_run)
print(no_decomposer_total_sum_run)
output_file_path = 'token_sums.csv'

with open(output_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Preference Index', 'Decomposer Total Sum', 'No Decomposer Total Sum'])
    for idx in range(20):
        writer.writerow([idx, decomposer_total_sum_run[idx], no_decomposer_total_sum_run[idx]])

print(f"Results written to {output_file_path}")

