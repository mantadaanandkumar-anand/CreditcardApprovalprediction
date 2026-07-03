# ==============================================================================
# Credit Card Approval Prediction System - Model Training Orchestrator
# ==============================================================================

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

# Machine Learning Modules
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

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
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    XGB_AVAILABLE = False


def load_and_inspect_data(dataset_path: Path) -> pd.DataFrame:
    """Loads credit card application data and prints diagnostics."""
    print("=== Loading Dataset ===")
    df = pd.read_csv(dataset_path)

    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    
    print("\n=== Head (First 5 Rows) ===")
    print(df.head())

    print("\n=== Tail (Last 5 Rows) ===")
    print(df.tail())

    print("\n=== Shape ===")
    print(df.shape)

    print("\n=== Columns ===")
    print(df.columns.tolist())

    print("\n=== Info ===")
    df.info()

    print("\n=== Describe (Summary Statistics) ===")
    print(df.describe(include="all"))

    print("\n=== Memory Usage ===")
    print(df.memory_usage(deep=True))

    print("\n=== Data Types ===")
    print(df.dtypes)
    
    return df


def run_eda(df: pd.DataFrame, figures_dir: Path, target_col: str = "approved"):
    """Runs exploratory data analysis and saves visualization charts."""
    print("\n=== Running Exploratory Data Analysis (EDA) ===")

    # 1. Missing Values
    missing_values = df.isnull().sum()
    print("\n--- Missing Values ---")
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found.")

    # 2. Duplicate Records
    duplicates_count = df.duplicated().sum()
    print(f"\nTotal duplicate records: {duplicates_count}")

    # 3. Unique Values per Column
    print("\n--- Unique Values per Column ---")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

    # Split columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # 4. Target Variable Distribution
    print(f"\n--- Target Variable ('{target_col}') Distribution ---")
    print(df[target_col].value_counts())
    print(df[target_col].value_counts(normalize=True))

    # 5. Outlier Detection using IQR Method
    print("\n--- Outlier Detection (IQR Method) ---")
    continuous_cols = [col for col in ['annual_income', 'age', 'years_employed', 'credit_score', 'debt_to_income'] if col in df.columns]
    for col in continuous_cols:
        temp_col = df[col].fillna(df[col].median())
        q1 = temp_col.quantile(0.25)
        q3 = temp_col.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"{col}: Found {len(outliers)} outliers (Bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")

    # Generate visual plots
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Correlation Matrix Heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_heatmap.png", dpi=140)
    plt.close()

    # Histograms (Annual Income)
    if 'annual_income' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x="annual_income", kde=True, bins=30, color="skyblue")
        plt.title("Distribution of Annual Income")
        plt.xlabel("Annual Income")
        plt.tight_layout()
        plt.savefig(figures_dir / "income_histogram.png", dpi=140)
        plt.close()

    # Countplots (Approval Status & Education Level)
    if target_col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=target_col, hue=target_col, palette="Set2", legend=False)
        plt.title("Distribution of Credit Approval Status")
        plt.xlabel("Approved (0 = Rejected, 1 = Approved)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(figures_dir / "approval_countplots.png", dpi=140)
        plt.close()

    if 'education' in df.columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, y="education", hue="education", palette="viridis", legend=False)
        plt.title("Education Level of Applicants")
        plt.xlabel("Count")
        plt.ylabel("Education Level")
        plt.tight_layout()
        plt.savefig(figures_dir / "education_countplots.png", dpi=140)
        plt.close()

    # Pie Chart (Gender Distribution)
    if 'gender' in df.columns:
        plt.figure(figsize=(6, 6))
        gender_counts = df['gender'].value_counts()
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['pink', 'lightblue'])
        plt.title("Gender Distribution of Applicants")
        plt.tight_layout()
        plt.savefig(figures_dir / "gender_pie_chart.png", dpi=140)
        plt.close()

    # Boxplots (Credit Score by Approval Status)
    if 'credit_score' in df.columns and target_col in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=target_col, y="credit_score", palette="pastel", hue=target_col, legend=False)
        plt.title("Credit Score by Approval Status")
        plt.xlabel("Approved (0 = Rejected, 1 = Approved)")
        plt.ylabel("Credit Score")
        plt.tight_layout()
        plt.savefig(figures_dir / "credit_score_boxplot.png", dpi=140)
        plt.close()

    # Pairplots
    pairplot_features = [col for col in ['age', 'annual_income', 'credit_score', target_col] if col in df.columns]
    if len(pairplot_features) > 1:
        pp = sns.pairplot(df[pairplot_features], hue=target_col if target_col in pairplot_features else None, palette="husl")
        pp.fig.suptitle("Pairplot of Key Financial Indicators", y=1.02)
        plt.tight_layout()
        pp.savefig(figures_dir / "key_features_pairplots.png", dpi=140)
        plt.close()

    print("EDA Complete. All visualization charts saved to:", figures_dir)


def split_dataset(df: pd.DataFrame, target_col: str = "approved") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the dataset into training (80%) and testing (20%) subsets."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split into 80% Train and 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=42, 
        stratify=y
    )

    print("\n=== Train-Test Split Complete ===")
    print(f"X_train Shape: {X_train.shape} | y_train Shape: {y_train.shape} (80%)")
    print(f"X_test Shape:  {X_test.shape}  | y_test Shape:  {y_test.shape}  (20%)")
    return X_train, X_test, y_train, y_test


def plot_model_comparison(comparison_df: pd.DataFrame, figures_dir: Path):
    """Generates comparison bar charts for the model benchmark metrics."""
    # Reshape DataFrame for plotting
    melted_df = comparison_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
                                   var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric", palette="Set1")
    plt.title("Classifier Models Benchmark Comparison")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    comp_chart_path = figures_dir / "model_comparison_bar_chart.png"
    plt.savefig(comp_chart_path, dpi=140)
    plt.close()
    print(f"Saved model comparison chart to: {comp_chart_path}")


def main():
    # Set paths
    dataset_path = project_root / "data" / "creditcard.csv"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        print(f"[ERROR] Dataset file not found at {dataset_path}. Please run generate_dataset.py first.")
        return

    # 1. Load and Inspect
    raw_df = load_and_inspect_data(dataset_path)

    # 2. Run EDA
    run_eda(raw_df, figures_dir)

    # 3. Data Cleaning
    cleaned_df = clean_data(raw_df)

    # 4. Feature Engineering
    engineered_df = add_engineered_features(cleaned_df)

    # 5. Feature Selection
    final_df = select_features(engineered_df)

    # 6. Train-Test Split
    X_train, X_test, y_train, y_test = split_dataset(final_df)

    # Get Preprocessing pipeline
    categorical_features = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
    numeric_features = X_train.select_dtypes(exclude=['object', 'string']).columns.tolist()
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    # Define the 5 model candidates
    if XGB_AVAILABLE:
        ensemble_clf = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss', use_label_encoder=False)
        ensemble_name = "XGBoost"
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        ensemble_clf = HistGradientBoostingClassifier(max_iter=150, max_depth=5, learning_rate=0.1, class_weight='balanced', random_state=42)
        ensemble_name = "HistGradientBoosting (Fallback)"

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=220, max_depth=9, class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42),
        ensemble_name: ensemble_clf
    }

    results = []
    trained_pipelines = {}

    print("\n==============================================================================")
    print(" TRAINING AND EVALUATING ALL CLASSIFIERS")
    print("==============================================================================")
    
    # 7. Training and Evaluating loop
    for name, clf in models.items():
        print(f"\n---> Training: {name}")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Predict
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC": auc
        })

        # Save individual model
        clean_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        model_save_path = models_dir / f"{clean_name}_model.joblib"
        joblib.dump(pipeline, model_save_path)
        print(f"Saved {name} Pipeline to: {model_save_path}")

    # 8. Comparison Table
    comparison_df = pd.DataFrame(results)
    print("\n==============================================================================")
    print(" MODEL PERFORMANCE BENCHMARK COMPARISON")
    print("==============================================================================")
    print(comparison_df.to_string(index=False))

    # Save comparison table
    comparison_df.to_csv(reports_dir / "model_comparison.csv", index=False)
    print(f"\nSaved model comparison metrics table to: {reports_dir / 'model_comparison.csv'}")

    # 9. Plot Model Comparison Metrics
    plot_model_comparison(comparison_df, figures_dir)

    # 10. Automatically select and serialize the Best Model
    # Sort models by ROC-AUC and F1 Score to select the best generalizer
    sorted_comparison = comparison_df.sort_values(by=["ROC AUC", "F1 Score"], ascending=False)
    best_model_name = sorted_comparison.iloc[0]["Model"]
    print(f"\n[BEST MODEL] Best Performing Model Identified: {best_model_name}")

    best_pipeline = trained_pipelines[best_model_name]
    
    # Save best model pipeline
    best_model_path = models_dir / "best_model.joblib"
    joblib.dump(best_pipeline, best_model_path)
    print(f"Saved Best Model Pipeline to: {best_model_path}")

    # Save model metadata object
    metadata_path = models_dir / "model_metadata.joblib"
    joblib.dump({
        "best_model_name": best_model_name,
        "features": X_train.columns.tolist(),
        "performance_benchmark": results
    }, metadata_path)
    print(f"Saved Model Metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
