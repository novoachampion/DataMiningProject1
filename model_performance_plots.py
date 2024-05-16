import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_learning_curve(epochs, accuracies, title):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracies, marker='o', linestyle='-', color='b')
    plt.title(f'Learning Curve for {title}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(f'performance/{title}_learning_curve.png')
    plt.show()

def plot_roc_curve(fpr, tpr, title):
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {title}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'performance/{title}_roc_curve.png')
    plt.show()

def plot_precision_recall_curve(precision, recall, title):
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, color='green', lw=2, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {title}')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(f'performance/{title}_precision_recall_curve.png')
    plt.show()

def main():
    # Example usage
    df = pd.read_csv('epochs/knn_metrics.csv')
    plot_learning_curve(df['epoch'], df['accuracy'], 'kNN')
    fpr, tpr, _ = roc_curve(df['actual'], df['probs'])
    plot_roc_curve(fpr, tpr, 'Decision Tree')
    precision, recall, _ = precision_recall_curve(df['actual'], df['probs'])
    plot_precision_recall_curve(precision, recall, 'GBM')

if __name__ == "__main__":
    main()
