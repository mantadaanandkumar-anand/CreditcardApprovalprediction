# Credit Card Approval Prediction System - Preprocessing
# Cleans raw application data, handles duplicates, performs missing values imputation, and handles feature engineering.
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs core data cleaning:
    1. Removes duplicate records.
    2. Standardizes column names (snake_case conversion).
    3. Fixes incorrect data types by casting flags and counts to integers.
    4. Caps outliers for numerical metrics using the IQR method.
    5. Standardizes string formatting for categorical features.
    """
    data = df.copy()
    
    # 1. Handle Duplicate Values
    before_dup = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    after_dup = len(data)
    print(f"Removed {before_dup - after_dup} duplicate records.")

    # 2. Column Renaming
    data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]
    
    # 3. Handle Incorrect Data Types
    # Cast binary indicators to integers
    flag_cols = [col for col in ['mobile_phone', 'work_phone', 'email'] if col in data.columns]
    for col in flag_cols:
        data[col] = data[col].fillna(0).astype(int)
    
    # Cast counting features, credit scores, and age to integer values
    int_cols = [col for col in ['age', 'children_count', 'family_members', 'credit_score', 'existing_loans'] if col in data.columns]
    for col in int_cols:
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val).round().astype(int)

    # 4. Outliers Handling (IQR Capping / Winsorization)
    # Caps features at Q1 - 1.5*IQR and Q3 + 1.5*IQR to prevent outliers from skewing predictive models
    continuous_cols = [col for col in ['annual_income', 'years_employed', 'debt_to_income'] if col in data.columns]
    for col in continuous_cols:
        temp_col = data[col].fillna(data[col].median())
        q1 = temp_col.quantile(0.25)
        q3 = temp_col.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data[col] = data[col].clip(lower_bound, upper_bound)
        print(f"Outliers in '{col}' capped at IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

    # 5. Feature Formatting (Categorical string normalization)
    # Strip whitespace, format to title case, and restore actual NaN values
    string_cols = data.select_dtypes(include=['object', 'string']).columns
    for col in string_cols:
        data[col] = data[col].astype(str).str.strip().str.title()
        data[col] = data[col].replace('Nan', np.nan)

    return data


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies custom feature engineering to generate relative indicators:
    - income_per_family_member: annual_income / family_members
    - employment_to_age_ratio: years_employed / age
    - high_credit_score: binary flag for credit score >= 700
    """
    data = df.copy()
    
    if 'annual_income' in data.columns and 'family_members' in data.columns:
        data['income_per_family_member'] = data['annual_income'] / data['family_members'].replace(0, 1)
        
    if 'years_employed' in data.columns and 'age' in data.columns:
        data['employment_to_age_ratio'] = data['years_employed'] / data['age'].replace(0, 1)
        
    if 'credit_score' in data.columns:
        data['high_credit_score'] = (data['credit_score'] >= 700).astype(int)
        
    return data


def get_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Creates a reusable scikit-learn ColumnTransformer preprocessor pipeline:
    - Numerical Pipeline: Imputes missing values with the median and normalizes with StandardScaler.
    - Categorical Pipeline: Imputes missing values with the most frequent value and encodes via OneHotEncoder.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature selection:
    1. Removes constant columns (zero variance, e.g., mobile_phone).
    2. Drops non-predictive or redundant features.
    """
    data = df.copy()
    
    # Identify constant columns (zero variance)
    # Variance is only calculated on sets with multiple records (e.g. during training)
    # For a single prediction row, we drop the specific known constant columns (mobile_phone)
    if len(data) > 1:
        constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
        if constant_cols:
            print(f"Feature Selection: Dropping constant features with zero variance: {constant_cols}")
            data = data.drop(columns=constant_cols)
    else:
        # During single-row prediction
        constant_cols = [col for col in ['mobile_phone'] if col in data.columns]
        if constant_cols:
            data = data.drop(columns=constant_cols)
        
    return data


def save_preprocessing_artifacts(preprocessor: ColumnTransformer, save_dir: str = "models") -> None:
    """
    Saves the fit preprocessor pipeline state to disk.
    This enables loading the scaler and encoders as a unit for deployment inference.
    """
    from pathlib import Path
    save_path = Path(save_dir) / "preprocessor.joblib"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, save_path)
    print(f"Saved fit preprocessor (scaler & encoders) to: {save_path}")

