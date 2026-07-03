# ==============================================================================
# Credit Card Approval Prediction System - XGBoost Model (with Fallback)
# ==============================================================================

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

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

# ------------------------------------------------------------------------------
# Dynamic Model Selection (XGBoost vs. HistGradientBoostingFallback)
# ------------------------------------------------------------------------------
try:
    import xgboost as xgb  # type: ignore
    XGB_AVAILABLE = True
    print("[INFO] XGBoost is available. Using XGBClassifier.")
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    XGB_AVAILABLE = False
    print("[WARNING] XGBoost is not installed in the environment.")
    print("[INFO] Falling back to Scikit-Learn's native HistGradientBoostingClassifier.")


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, preprocessor) -> Pipeline:
    """
    Creates and trains a pipeline combining the preprocessor and the chosen classifier.
    
    If XGBoost is available, trains XGBClassifier. Otherwise, falls back to
    HistGradientBoostingClassifier.
    """
    print("\n--- Training Ensemble Model ---")
    
    if XGB_AVAILABLE:
        # Define XGBoost model
        # Use scale_pos_weight if needed to balance classes, or standard weights
        clf = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
    else:
        # Define HistGradientBoosting model (highly optimized native sklearn fallback)
        clf = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=5,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42
        )
        
    # Create the training pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
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
    plt.title('Ensemble Model Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = figures_dir / 'xgb_confusion_matrix.png'
    plt.savefig(cm_path, dpi=140)
    plt.close()
    print(f"\nSaved Confusion Matrix Plot to: {cm_path}")
    
    # Plot & Save ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='teal', lw=2, label=f'ROC curve (AUC = {metrics["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Ensemble Model ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = figures_dir / 'xgb_roc_curve.png'
    plt.savefig(roc_path, dpi=140)
    plt.close()
    print(f"Saved ROC Curve Plot to: {roc_path}")
    
    return metrics


def display_feature_importance(pipeline: Pipeline, X_train: pd.DataFrame):
    """
    Retrieves, lists, and visualizes the feature importances if available.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    # Try to retrieve column feature names after transformer pipeline runs
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"Feature_{i}" for i in range(len(X_train.columns))]
        
    # Get feature importances if the model supports it
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'permutation_importance'):
        # Fallback for models without direct feature_importances_ attribute
        importances = np.zeros(len(feature_names))
    else:
        # Some models (like HistGradientBoostingClassifier) require permutation importance
        from sklearn.inspection import permutation_importance
        # Compute permutation importance on a small subset for speed
        # To avoid computing during clean imports, we write a fallback warning
        print("\n[INFO] Feature importance plotting is not directly supported by this classifier type without fitting permutation importance. Skipping plotting.")
        return
        
    # Create DataFrame for analysis
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Display top 15 features
    top_features = feat_imp.head(15)
    print("\n=== Feature Importances (Top 15) ===")
    print(top_features.to_string(index=False))
    
    # Plot & Save Feature Importance chart
    figures_dir = project_root / "reports" / "figures"
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_features, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title('Ensemble Model Feature Importances')
    plt.xlabel('Relative Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    imp_path = figures_dir / 'xgb_feature_importance.png'
    plt.savefig(imp_path, dpi=140)
    plt.close()
    print(f"Saved Feature Importance Bar Chart to: {imp_path}")


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
    model_pipeline = train_xgboost(X_train, y_train, preprocessor)
    
    # Evaluate model
    evaluate_model(model_pipeline, X_test, y_test)
    
    # Display and Save Feature Importance (only if using XGBoost, since HistGradientBoosting requires permutation_importance)
    if XGB_AVAILABLE:
        display_feature_importance(model_pipeline, X_train)
    
    # Save the complete pipeline
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "xgboost_model.joblib"
    joblib.dump(model_pipeline, model_path)
    print(f"Saved trained Ensemble Pipeline to: {model_path}")
    
    # Extract and save individual components: model.pkl, scaler.pkl, encoder.pkl
    classifier = model_pipeline.named_steps['classifier']
    preprocessor = model_pipeline.named_steps['preprocessor']
    scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
    encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    
    joblib.dump(classifier, models_dir / "model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(encoder, models_dir / "encoder.pkl")
    print("Saved model.pkl, scaler.pkl, and encoder.pkl successfully to models/ directory.")
    
    return model_pipeline

if __name__ == "__main__":
    main()
