import pandas as pd
from data_preparation import preprocess_data
from gbm_model import run_gbm
from decision_tree_model import run_decision_tree
from knn_model import run_knn

def main():
    filepath = 'data/DATA.csv'
    df = pd.read_csv(filepath)

    # Preprocess the data
    df_preprocessed, _ = preprocess_data(df)
    
    # Run GBM model
    gbm_metrics = run_gbm(df_preprocessed)
    print("GBM Model Metrics:")
    for _, row in gbm_metrics.iterrows():
        print(f"Epoch {row['epoch']}: Accuracy: {row['accuracy']:.4f}, Precision: {row['precision']:.4f}, Recall: {row['recall']:.4f}, F1-Score: {row['f1_score']:.4f}, ROC AUC: {row['roc_auc']:.4f}")

    # Run Decision Tree model
    dt_metrics = run_decision_tree(df_preprocessed)
    print("\nDecision Tree Model Metrics:")
    for _, row in dt_metrics.iterrows():
        print(f"Epoch {row['epoch']}: Accuracy: {row['accuracy']:.4f}, Precision: {row['precision']:.4f}, Recall: {row['recall']:.4f}, F1-Score: {row['f1_score']:.4f}, ROC AUC: {row['roc_auc']:.4f}")

    # Run kNN model
    knn_metrics = run_knn(df_preprocessed)
    print("\nkNN Model Metrics:")
    for _, row in knn_metrics.iterrows():
        print(f"Epoch {row['epoch']}: Accuracy: {row['accuracy']:.4f}, Precision: {row['precision']:.4f}, Recall: {row['recall']:.4f}, F1-Score: {row['f1_score']:.4f}, ROC AUC: {row['roc_auc']:.4f}")

if __name__ == "__main__":
    main()
