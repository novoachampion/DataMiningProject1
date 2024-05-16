import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

def run_decision_tree(df):
    X = df.drop('Grade', axis=1)
    y = df['Grade']
    y_binarized = label_binarize(y, classes=sorted(y.unique()))

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_test_binarized = label_binarize(y_test, classes=sorted(y.unique()))
    
    clf = DecisionTreeClassifier(random_state=42)
    metrics_history = []

    for epoch in range(1, 9):  # Assuming 8 epochs
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        roc_auc = roc_auc_score(y_test_binarized, y_proba, average='macro')
        
        metrics_history.append({
            'epoch': epoch,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'actual': y_test.tolist(),
            'probs': y_proba.tolist()
        })
    
    # Create directory if it does not exist
    os.makedirs('epochs', exist_ok=True)
    
    # Convert to DataFrame and save to CSV
    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.to_csv('epochs/decision_tree_metrics.csv', index=False)

    return metrics_df

# Example usage
if __name__ == "__main__":
    df = pd.read_csv('data/DATA.csv')
    metrics = run_decision_tree(df)
    print(metrics)
