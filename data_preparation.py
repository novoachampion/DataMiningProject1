# data_preparation.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Drop 'StudentID' as it's just an identifier
    if 'StudentID' in df.columns:
        df = df.drop('StudentID', axis=1)

    # Encode all categorical variables
    label_encoders = {}
    for column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    return df, label_encoders
