import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from pathlib import Path

def cramers_v(x, y, small_value=1e-10):
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2)/(n-1) + small_value
    k_corr = k - ((k-1)**2)/(n-1) + small_value
    return np.sqrt(phi2_corr / min((k_corr-1), (r_corr-1)))

def generate_correlation_matrix(filepath, output_dir):
    print("Starting the script...")

    df = pd.read_csv(filepath)
    print("Data loaded successfully. Processing correlation matrix...")

    columns = df.columns
    corr_matrix = pd.DataFrame(np.zeros((len(columns), len(columns))), index=columns, columns=columns)
    
    for col1 in columns:
        for col2 in columns:
            corr_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

    print("Correlation matrix calculated based on Cramér's V.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filepath = output_dir / 'correlation_matrix.xlsx'
    corr_matrix.to_excel(output_filepath)
    print(f"Correlation matrix saved to {output_filepath}")

if __name__ == "__main__":
    data_filepath = Path('data/DATA.csv')
    output_directory = Path('processedData')
    generate_correlation_matrix(data_filepath, output_directory)
