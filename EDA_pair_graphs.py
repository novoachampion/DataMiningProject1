import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data from the CSV file
input_path = "./data/DATA.csv"
data = pd.read_csv(input_path)

# Define labels for each category and grades
labels = {
    'StudentAge': {1: '18-21', 2: '22-25', 3: 'above 26'},
    'Sex': {1: 'female', 2: 'male'},
    'SchoolType': {1: 'private', 2: 'state', 3: 'other'},
    'Scholarship': {1: 'None', 2: '25%', 3: '50%', 4: '75%', 5: 'Full'},
    'Work': {1: 'Yes', 2: 'No'},
    'ArtActivity': {1: 'Yes', 2: 'No'},
    'Partner': {1: 'Yes', 2: 'No'},
    'Salary': {1: 'USD 135-200', 2: 'USD 201-270', 3: 'USD 271-340', 4: 'USD 341-410', 5: 'above 410'},
    'Transport': {1: 'Bus', 2: 'Private car/taxi', 3: 'bicycle', 4: 'Other'},
    'Accommodation': {1: 'rental', 2: 'dormitory', 3: 'with family', 4: 'Other'},
    'MotherEd': {1: 'primary school', 2: 'secondary school', 3: 'high school', 4: 'university', 5: 'MSc.', 6: 'Ph.D.'},
    'FatherEd': {1: 'primary school', 2: 'secondary school', 3: 'high school', 4: 'university', 5: 'MSc.', 6: 'Ph.D.'},
    'Siblings': {1: '1', 2: '2', 3: '3', 4: '4', 5: '5 or above'},
    'ParentalStatus': {1: 'married', 2: 'divorced', 3: 'died - one of them or both'},
    'MotherOcc': {1: 'retired', 2: 'housewife', 3: 'government officer', 4: 'private sector employee', 5: 'self-employment', 6: 'other'},
    'FatherOcc': {1: 'retired', 2: 'government officer', 3: 'private sector employee', 4: 'self-employment', 5: 'other'},
    'StudyHours': {1: 'None', 2: '<5 hours', 3: '6-10 hours', 4: '11-20 hours', 5: 'more than 20 hours'},
    'ReadingNonSci': {1: 'None', 2: 'Sometimes', 3: 'Often'},
    'ReadingSci': {1: 'None', 2: 'Sometimes', 3: 'Often'},
    'SeminarAttendance': {1: 'Yes', 2: 'No'},
    'ProjectImpact': {1: 'positive', 2: 'negative', 3: 'neutral'},
    'ClassAttendance': {1: 'always', 2: 'sometimes', 3: 'never'},
    'MidtermPrep1': {1: 'alone', 2: 'with friends', 3: 'not applicable'},
    'MidtermPrep2': {1: 'closest date to the exam', 2: 'regularly during the semester', 3: 'never'},
    'NoteTaking': {1: 'never', 2: 'sometimes', 3: 'always'},
    'ClassListening': {1: 'never', 2: 'sometimes', 3: 'always'},
    'DiscussionImpact': {1: 'never', 2: 'sometimes', 3: 'always'},
    'FlipClassroom': {1: 'not useful', 2: 'useful', 3: 'not applicable'},
    'LastGPA': {1: '<2.00', 2: '2.00-2.49', 3: '2.50-2.99', 4: '3.00-3.49', 5: 'above 3.49'},
    'ExpectedGPA': {1: '<2.00', 2: '2.00-2.49', 3: '2.50-2.99', 4: '3.00-3.49', 5: 'above 3.49'},
    'Grade': {0: 'Fail', 1: 'DD', 2: 'DC', 3: 'CC', 4: 'CB', 5: 'BB', 6: 'BA', 7: 'AA'}
}

# Ensure the output directory exists
output_directory = './grades_comparison'
os.makedirs(output_directory, exist_ok=True)

# Iterate over each category, excluding the 'Grade' column
for col in data.columns[:-1]:  # Assuming 'Grade' is the last column
    if col in labels:
        # Create figure
        plt.figure(figsize=(10, 6))
        # Generate bar plot comparing each category with 'Grade'
        pd.crosstab(data[col], data['Grade']).plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        plt.title(f'Comparison of {col} with Grades')
        plt.xlabel('Category')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.legend([f'{grade}: {labels["Grade"][grade]}' for grade in sorted(labels['Grade'].keys())], title='Grade', bbox_to_anchor=(1.05, 1), loc='upper left')
        # Save plot
        plt.savefig(f'{output_directory}/{col}_vs_grade.png', bbox_inches='tight')
        plt.close()

        # Print counts to console for each category
        count_data = pd.crosstab(data[col], data['Grade'])
        print(f'Counts for {col}:')
        print(count_data)
        print('\n')
