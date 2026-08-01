import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    print("--- DecodeLabs Project 2: Data Classification Pipeline ---")
    
    # STEP 1: Load and Understand the Iris Benchmark Dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"\n[1] DATASET OVERVIEW")
    print(f"Total Samples: {X.shape[0]}")
    print(f"Features ({X.shape[1]}): {feature_names}")
    print(f"Classes ({len(target_names)}): {target_names}")

    # STEP 2: Structural Split (80% Training, 20% Testing) with Stratification/Shuffle
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[2] DATA SPLIT COMPLETE")
    print(f"Training samples: {X_train.shape[0]} (80%)")
    print(f"Testing samples:  {X_test.shape[0]} (20%)")

    # STEP 3: Feature Scaling (StandardScaler: Mean = 0, Variance = 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\n[3] FEATURE SCALING APPLIED (StandardScaler)")

    # STEP 4: Algorithm Model - K-Nearest Neighbors (KNN with K = 5)
    k_value = 5
    model = KNeighborsClassifier(n_neighbors=k_value)
    
    # FIT (Memorize the map)
    model.fit(X_train_scaled, y_train)
    print(f"\n[4] MODEL TRAINED (KNN Classifier, K={k_value})")

    # PREDICT (Apply logic to test set)
    y_pred = model.predict(X_test_scaled)

    # STEP 5: Output Validation & Diagnostics
    print("\n[5] MODEL EVALUATION RESULTS")
    print("-" * 50)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {acc * 100:.2f}%\n")

    print("--- Diagnostic Tool: Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\n--- Strategic Metrics: Precision, Recall, & F1-Score ---")
    print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == "__main__":
    main()
