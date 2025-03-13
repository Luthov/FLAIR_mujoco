import csv
import ast
import numpy as np
import matplotlib.pyplot as plt

def open_csv_file(file_path):
    satisfaction_a = []
    satisfaction_b = []
    preferences_a = []
    preferences_b = []
    safety_a = []
    safety_b = []
    comfort_a = []
    comfort_b = []
    responsiveness_a = []
    responsiveness_b = []
    ease_a = [] 
    ease_b = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            satisfaction_a.append(ast.literal_eval(row[0]))
            satisfaction_b.append(ast.literal_eval(row[7]))
            preferences_a.append(ast.literal_eval(row[1]))
            preferences_b.append(ast.literal_eval(row[8]))
            safety_a.append(ast.literal_eval(row[2]))
            safety_b.append(ast.literal_eval(row[9]))
            comfort_a.append(ast.literal_eval(row[3]))
            comfort_b.append(ast.literal_eval(row[10]))
            responsiveness_a.append(ast.literal_eval(row[4]))
            responsiveness_b.append(ast.literal_eval(row[11]))
            ease_a.append(ast.literal_eval(row[5]))
            ease_b.append(ast.literal_eval(row[12]))

    return (satisfaction_a, satisfaction_b, preferences_a, preferences_b, 
            safety_a, safety_b, comfort_a, comfort_b, 
            responsiveness_a, responsiveness_b, ease_a, ease_b)

def plot_side_by_side(participants, data_a, data_b, title, ylabel, filename):
    x = np.arange(len(participants))  # X locations for the groups
    width = 0.3  # Width of the bars

    fig, ax = plt.subplots()

    # Plot bars
    bars1 = ax.bar(x - width/2, data_a, width, label='Scenario A', color='b')
    bars2 = ax.bar(x + width/2, data_b, width, label='Scenario B', color='g')

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(participants)
    ax.set_xlabel('Participants')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    plt.savefig(filename)
    plt.close()

# Example usage
file_path = 'Scenario_A_B.csv'
(satisfaction_a, satisfaction_b, preferences_a, preferences_b, 
 safety_a, safety_b, comfort_a, comfort_b, 
 responsiveness_a, responsiveness_b, ease_a, ease_b) = open_csv_file(file_path)

participants = ['Chris', 'Ben', 'Aaradh', 'Vassanth', 'Ritesh', 'Gabriel', 'Hauwen']

# Save each plot
plot_side_by_side(participants, satisfaction_a, satisfaction_b, 'Satisfaction Rating', 'Satisfaction', 'satisfaction_rating.png')
plot_side_by_side(participants, preferences_a, preferences_b, 'Preferences Rating', 'Preferences', 'preferences_rating.png')
plot_side_by_side(participants, safety_a, safety_b, 'Safety Rating', 'Safety', 'safety_rating.png')
plot_side_by_side(participants, comfort_a, comfort_b, 'Comfort Rating', 'Comfort', 'comfort_rating.png')
plot_side_by_side(participants, responsiveness_a, responsiveness_b, 'Responsiveness Rating', 'Responsiveness', 'responsiveness_rating.png')
plot_side_by_side(participants, ease_a, ease_b, 'Ease Rating', 'Ease', 'ease_rating.png')
