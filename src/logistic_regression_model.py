# ==============================================================================
# Credit Card Approval Prediction System - Logistic Regression Model
# ==============================================================================

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import train_test_split

# Add project root to path for modular imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing import clean_data, add_engineered_features, select_features, get_preprocessor


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, preprocessor) -> Pipeline:
    """
    Creates and trains a pipeline combining the preprocessor and a Logistic Regression classifier.
    """
    print("\n--- Training Logistic Regression Model ---")
    
    # Instantiate Logistic Regression with class weighting to handle potential imbalances
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    
    # Create the training pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lr)
    ])
    
    # Fit the pipeline on training data
    pipeline.fit(X_train, y_train)
    print("Training Completed.")
    
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluates the model on test data, prints metrics, and saves evaluation figures.
    """
    print("\n=== Model Evaluation (Test Data) ===")
    
    # Predict classes and probabilities
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    # Print metrics
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    
    # Print Classification Report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Calculate and print Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("=== Confusion Matrix ===")
    print(cm)
    
    # Ensure visualization figures output directory exists
    figures_dir = project_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot & Save Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'])
    plt.title('Logistic Regression Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = figures_dir / 'lr_confusion_matrix.png'
    plt.savefig(cm_path, dpi=140)
    plt.close()
    print(f"\nSaved Confusion Matrix Plot to: {cm_path}")
    
    # Plot & Save ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = figures_dir / 'lr_roc_curve.png'
    plt.savefig(roc_path, dpi=140)
    plt.close()
    print(f"Saved ROC Curve Plot to: {roc_path}")
    
    return metrics


def main():
    # Load dataset
    dataset_path = project_root / "data" / "creditcard.csv"
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}. Please run generate_dataset.py first.")
        return
        
    df = pd.read_csv(dataset_path)
    
    # Clean data
    cleaned_df = clean_data(df)
    
    # Feature Engineering
    engineered_df = add_engineered_features(cleaned_df)
    
    # Feature Selection
    final_df = select_features(engineered_df)
    
    target_col = 'approved'
    X = final_df.drop(columns=[target_col])
    y = final_df[target_col]
    
    # Train-test split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Define features for ColumnTransformer
    categorical_features = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
    numeric_features = X_train.select_dtypes(exclude=['object', 'string']).columns.tolist()
    
    # Get preprocessor pipeline
    preprocessor = get_preprocessor(numeric_features, categorical_features)
    
    # Train the pipeline
    model_pipeline = train_logistic_regression(X_train, y_train, preprocessor)
    
    # Evaluate model
    evaluate_model(model_pipeline, X_test, y_test)
    
    # Save the complete pipeline (includes fits for scalers, encoders, and estimator weights)
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "logistic_regression_model.joblib"
    joblib.dump(model_pipeline, model_path)
    print(f"Saved trained Logistic Regression Pipeline to: {model_path}")
    
    return model_pipeline

if __name__ == "__main__":
    main()
