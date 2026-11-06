import argparse
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data", "Cancer-classification-dataset")

parser = argparse.ArgumentParser()
parser.add_argument("--cancer", required=True, choices=["BRCA", "COAD", "GBM", "LGG", "OV"])
parser.add_argument("--data_path", default=DEFAULT_DATA_DIR, help="Path to folder containing graph file")
args = parser.parse_args()

def find_graph_path():
    candidates = [
        args.data_path,
        os.path.join(os.getcwd(), args.data_path) if args.data_path else None,
        DEFAULT_DATA_DIR,
        os.path.join(os.getcwd(), "data", "Cancer-classification-dataset"),
        os.getcwd(),
        BASE_DIR,
    ]
    for c in candidates:
        if not c:
            continue
        if c.endswith(".pt") and os.path.exists(c):
            return c
        target = os.path.join(c, f"{args.cancer}_graph.pt")
        if os.path.exists(target):
            return target
    return None

def run_baselines():
    print(f"\n==================================================")
    print(f"Running Baselines for {args.cancer}")
    print(f"==================================================")

    # Load graph data
    path = find_graph_path()
    if path is None or not os.path.exists(path):
        print(f"Error: Could not find '{args.cancer}_graph.pt'. Please run prepare_graph.py first.")
        return

    data = torch.load(path, weights_only=False)
    data_dir = os.path.dirname(path)
    
    # Extract concatenated features
    feature_list = [data['mRNA'].x.numpy(), data['miRNA'].x.numpy(), data['Methy'].x.numpy()]
    if 'CNV' in data:
        feature_list.append(data['CNV'].x.numpy())
    
    X = np.concatenate(feature_list, axis=1)
    y = data['mRNA'].y.numpy()
    
    train_mask = data['mRNA'].train_mask.numpy()
    test_mask = data['mRNA'].test_mask.numpy()
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Concatenated Feature Dimension: {X.shape[1]}")
    
    models = {
        'SVM': SVC(kernel='rbf', class_weight='balanced', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }

    log_path = os.path.join(data_dir, "results_log.txt")
    
    with open(log_path, "a") as f:
        f.write(f"\n{'='*55}\n")
        f.write(f"BASELINE MODELS - Cancer: {args.cancer}\n")
        f.write(f"{'='*55}\n")

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='macro', zero_division=0)
        
        counts = np.bincount(y, minlength=len(np.unique(y)))
        cls_names = [f"Class{i}({counts[i]})" for i in range(len(counts))]
        
        report = classification_report(y_test, pred, target_names=cls_names, zero_division=0)
        
        print(f"{name} Accuracy: {acc:.4f}")
        print(f"{name} Macro-F1: {f1:.4f}")
        
        with open(log_path, "a") as f:
            f.write(f"Model: {name}\n")
            f.write(f"Accuracy : {acc:.4f}\n")
            f.write(f"Macro-F1 : {f1:.4f}\n")
            f.write("Per-class report:\n")
            f.write(report)
            f.write(f"{'-'*55}\n")

if __name__ == "__main__":
    run_baselines()
