# ==============================================================================
# Credit Card Approval Prediction System - Hyperparameter Tuning
# ==============================================================================

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add project root to path for modular imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing import clean_data, add_engineered_features, select_features, get_preprocessor


def run_hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series, preprocessor) -> RandomForestClassifier:
    """
    Executes a two-stage tuning workflow for the Random Forest Classifier:
    1. RandomizedSearchCV: Broad, coarse search to locate promising parameter space.
    2. GridSearchCV: Localized, fine-grained search centered around RandomizedSearch results.
    """
    # Define baseline base estimator inside a pipeline step
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    # We define the pipeline structure
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])

    # --------------------------------------------------------------------------
    # STAGE 1: Randomized Search (Broad Parameters Grid)
    # --------------------------------------------------------------------------
    print("\n==============================================================")
    print(" STAGE 1: Broad Hyperparameter Search using RandomizedSearchCV")
    print("==============================================================")
    
    # Parameter prefixes match the pipeline step name ('classifier__')
    random_param_distributions = {
        'classifier__n_estimators': [int(x) for x in np.linspace(100, 500, 5)],
        'classifier__max_depth': [None] + [int(x) for x in np.linspace(5, 25, 5)],
        'classifier__min_samples_split': [2, 5, 10, 15],
        'classifier__min_samples_leaf': [1, 2, 4, 8],
        'classifier__max_features': ['sqrt', 'log2', None],
        'classifier__bootstrap': [True, False]
    }
    
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=random_param_distributions,
        n_iter=15,  # Number of random parameter combinations to sample
        cv=3,       # 3-Fold Cross-Validation splits
        scoring='roc_auc',  # Optimize for ROC AUC metric
        random_state=42,
        n_jobs=-1,  # Use all available CPU cores
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    print("\nRandomizedSearch Completed.")
    print("Best RandomizedSearch ROC-AUC Score:", f"{random_search.best_score_:.4f}")
    print("Best RandomizedSearch Params:")
    for k, v in random_search.best_params_.items():
        print(f"  {k}: {v}")

    # Extract the best parameters from RandomizedSearchCV to center our GridSearch
    best_rand = random_search.best_params_
    
    # --------------------------------------------------------------------------
    # STAGE 2: Grid Search (Tight Fine-Tuning Grid)
    # --------------------------------------------------------------------------
    print("\n==============================================================")
    print(" STAGE 2: Localized Fine-Tuning using GridSearchCV")
    print("==============================================================")
    
    # Construct a narrow grid around the best values found in stage 1
    # Example: If best n_estimators was 200, search [180, 200, 220]
    best_n_est = best_rand['classifier__n_estimators']
    best_depth = best_rand['classifier__max_depth']
    best_split = best_rand['classifier__min_samples_split']
    best_leaf = best_rand['classifier__min_samples_leaf']

    # Handle numeric grids safely
    n_estimators_grid = [max(50, best_n_est - 50), best_n_est, best_n_est + 50]
    
    if best_depth is None:
        max_depth_grid = [None, 20, 30]
    else:
        max_depth_grid = [max(3, best_depth - 3), best_depth, best_depth + 3]

    min_samples_split_grid = list(set([max(2, best_split - 2), best_split, best_split + 2]))
    min_samples_leaf_grid = list(set([max(1, best_leaf - 1), best_leaf, best_leaf + 1]))

    grid_params = {
        'classifier__n_estimators': n_estimators_grid,
        'classifier__max_depth': max_depth_grid,
        'classifier__min_samples_split': min_samples_split_grid,
        'classifier__min_samples_leaf': min_samples_leaf_grid,
        'classifier__max_features': [best_rand['classifier__max_features']],
        'classifier__bootstrap': [best_rand['classifier__bootstrap']]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid_params,
        cv=3,       # 3-Fold Cross Validation
        scoring='roc_auc',  # Focus optimization on ROC AUC
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print("\nGridSearch Completed.")
    print("Best GridSearch ROC-AUC Score:", f"{grid_search.best_score_:.4f}")
    print("Best Fine-Tuned Params:")
    for k, v in grid_search.best_params_.items():
        print(f"  {k}: {v}")
        
    return grid_search.best_estimator_


def main():
    # Load dataset
    dataset_path = project_root / "data" / "creditcard.csv"
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found at {dataset_path}. Please run generate_dataset.py first.")
        return
        
    raw_df = pd.read_csv(dataset_path)
    
    # Run Preprocessing Pipeline
    cleaned_df = clean_data(raw_df)
    engineered_df = add_engineered_features(cleaned_df)
    final_df = select_features(engineered_df)
    
    # Split Data
    X = final_df.drop(columns=['approved'])
    y = final_df['approved']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Configure Column Transformer
    categorical_features = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
    numeric_features = X_train.select_dtypes(exclude=['object', 'string']).columns.tolist()
    preprocessor = get_preprocessor(numeric_features, categorical_features)
    
    # Evaluate Baseline Model first
    baseline_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])
    baseline_pipeline.fit(X_train, y_train)
    y_pred_base = baseline_pipeline.predict(X_test)
    y_proba_base = baseline_pipeline.predict_proba(X_test)[:, 1]
    
    # Run Search Tuning
    tuned_pipeline = run_hyperparameter_tuning(X_train, y_train, preprocessor)
    
    # Evaluate Tuned Model
    y_pred_tuned = tuned_pipeline.predict(X_test)
    y_proba_tuned = tuned_pipeline.predict_proba(X_test)[:, 1]
    
    # Performance benchmark logs comparison
    print("\n==============================================================")
    print(" PERFORMANCE ADVANCEMENT EVALUATION (BASELINE vs. TUNED)")
    print("==============================================================")
    
    metrics_comparison = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
        "Baseline": [
            accuracy_score(y_test, y_pred_base),
            precision_score(y_test, y_pred_base, zero_division=0),
            recall_score(y_test, y_pred_base, zero_division=0),
            f1_score(y_test, y_pred_base, zero_division=0),
            roc_auc_score(y_test, y_proba_base)
        ],
        "Fine-Tuned": [
            accuracy_score(y_test, y_pred_tuned),
            precision_score(y_test, y_pred_tuned, zero_division=0),
            recall_score(y_test, y_pred_tuned, zero_division=0),
            f1_score(y_test, y_pred_tuned, zero_division=0),
            roc_auc_score(y_test, y_proba_tuned)
        ]
    })
    
    # Compute performance gain
    metrics_comparison["Gain"] = metrics_comparison["Fine-Tuned"] - metrics_comparison["Baseline"]
    print(metrics_comparison.to_string(index=False))
    
    # Save the optimized model pipeline
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    tuned_model_path = models_dir / "tuned_best_model.joblib"
    joblib.dump(tuned_pipeline, tuned_model_path)
    print(f"\nSaved Tuned Best Model Pipeline to: {tuned_model_path}")


if __name__ == "__main__":
    main()
