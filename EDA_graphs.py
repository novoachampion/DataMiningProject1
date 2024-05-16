import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data from the CSV file
input_path = "./data/DATA.csv"
data = pd.read_csv(input_path)

# Define labels for each category, shifted to match actual data columns correctly
labels = {
    2: {1: '18-21', 2: '22-25', 3: 'above 26'},
    3: {1: 'female', 2: 'male'},
    4: {1: 'private', 2: 'state', 3: 'other'},
    5: {1: 'None', 2: '25%', 3: '50%', 4: '75%', 5: 'Full'},
    6: {1: 'Yes', 2: 'No'},
    7: {1: 'Yes', 2: 'No'},
    8: {1: 'Yes', 2: 'No'},
    9: {1: 'USD 135-200', 2: 'USD 201-270', 3: 'USD 271-340', 4: 'USD 341-410', 5: 'above 410'},
    10: {1: 'Bus', 2: 'Private car/taxi', 3: 'bicycle', 4: 'Other'},
    11: {1: 'rental', 2: 'dormitory', 3: 'with family', 4: 'Other'},
    12: {1: 'primary school', 2: 'secondary school', 3: 'high school', 4: 'university', 5: 'MSc.', 6: 'Ph.D.'},
    13: {1: 'primary school', 2: 'secondary school', 3: 'high school', 4: 'university', 5: 'MSc.', 6: 'Ph.D.'},
    14: {1: '1', 2: '2', 3: '3', 4: '4', 5: '5 or above'},
    15: {1: 'married', 2: 'divorced', 3: 'died - one of them or both'},
    16: {1: 'retired', 2: 'housewife', 3: 'government officer', 4: 'private sector employee', 5: 'self-employment', 6: 'other'},
    17: {1: 'retired', 2: 'government officer', 3: 'private sector employee', 4: 'self-employment', 5: 'other'},
    18: {1: 'None', 2: '<5 hours', 3: '6-10 hours', 4: '11-20 hours', 5: 'more than 20 hours'},
    19: {1: 'None', 2: 'Sometimes', 3: 'Often'},
    20: {1: 'None', 2: 'Sometimes', 3: 'Often'},
    21: {1: 'Yes', 2: 'No'},
    22: {1: 'positive', 2: 'negative', 3: 'neutral'},
    23: {1: 'always', 2: 'sometimes', 3: 'never'},
    24: {1: 'alone', 2: 'with friends', 3: 'not applicable'},
    25: {1: 'closest date to the exam', 2: 'regularly during the semester', 3: 'never'},
    26: {1: 'never', 2: 'sometimes', 3: 'always'},
    27: {1: 'never', 2: 'sometimes', 3: 'always'},
    28: {1: 'never', 2: 'sometimes', 3: 'always'},
    29: {1: 'not useful', 2: 'useful', 3: 'not applicable'},
    30: {1: '<2.00', 2: '2.00-2.49', 3: '2.50-2.99', 4: '3.00-3.49', 5: 'above 3.49'},
    31: {1: '<2.00', 2: '2.00-2.49', 3: '2.50-2.99', 4: '3.00-3.49', 5: 'above 3.49'},
    32: {0: 'Fail', 1: 'DD', 2: 'DC', 3: 'CC', 4: 'CB', 5: 'BB', 6: 'BA', 7: 'AA'}
}

# Create a directory for the plots if it doesn't exist
output_directory = './plots'
os.makedirs(output_directory, exist_ok=True)

# Plot bar graphs for each category and save them
for col in range(2, 33):  # Start from 2 to skip 'StudentID'
    if col in labels:
        fig, ax = plt.subplots(figsize=(8, 6))
        counts = data.iloc[:, col - 1].value_counts().sort_index()
        counts.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title(f'Category {col - 1}: {data.columns[col - 1]}')
        ax.set_xlabel('Category')
        ax.set_ylabel('Frequency')
        ax.set_xticks(range(len(labels[col])))
        ax.set_xticklabels([labels[col][x] for x in sorted(labels[col].keys())], rotation=45)
        for i, v in enumerate(counts):
            ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
        plt.tight_layout()
        fig.savefig(f'{output_directory}/Category_{col - 1}_{data.columns[col - 1]}.png')
        plt.close(fig)

print("All plots have been saved.")
