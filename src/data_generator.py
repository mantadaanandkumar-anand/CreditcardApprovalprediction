# Credit Card Approval Prediction System - Data Generator
# Generates realistic synthetic credit application records with duplicates and missing values.

import numpy as np
import pandas as pd
from pathlib import Path

def generate_synthetic_data(output_path: Path, num_samples: int = 1500, random_seed: int = 42):
    np.random.seed(random_seed)
    
    # 1. Generate demographic columns
    gender = np.random.choice(['Male', 'Female'], size=num_samples, p=[0.45, 0.55])
    owns_car = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.38, 0.62])
    owns_property = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.65, 0.35])
    
    income_types = ['Working', 'Commercial associate', 'Pensioner', 'State servant']
    income_type = np.random.choice(income_types, size=num_samples, p=[0.55, 0.22, 0.15, 0.08])
    
    education_levels = ['Secondary / special education', 'Higher education', 'Incomplete higher', 'Lower secondary']
    education = np.random.choice(education_levels, size=num_samples, p=[0.70, 0.22, 0.06, 0.02])
    
    family_statuses = ['Married', 'Single', 'Civil marriage', 'Separated', 'Widow']
    family_status = np.random.choice(family_statuses, size=num_samples, p=[0.68, 0.16, 0.08, 0.06, 0.02])
    
    housing_types = ['House / apartment', 'Rented apartment', 'With parents', 'Municipal apartment']
    housing_type = np.random.choice(housing_types, size=num_samples, p=[0.88, 0.05, 0.05, 0.02])
    
    # 2. Generate numeric columns
    age = np.random.randint(18, 75, size=num_samples)
    
    # years_employed must be logically bounded by age
    years_employed = []
    for a in age:
        max_work = max(0, a - 18)
        years_employed.append(round(np.random.uniform(0, max_work) * 0.75, 1))
    years_employed = np.array(years_employed)
    
    annual_income = np.random.lognormal(mean=10.8, sigma=0.4, size=num_samples).round(-2)
    children_count = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.70, 0.18, 0.10, 0.02])
    family_members = children_count + np.random.choice([1, 2], size=num_samples, p=[0.20, 0.80])
    
    credit_score = np.random.randint(400, 850, size=num_samples)
    existing_loans = np.random.randint(0, 6, size=num_samples)
    debt_to_income = np.random.beta(a=2, b=5, size=num_samples) * 0.8
    
    mobile_phone = np.ones(num_samples, dtype=int)
    work_phone = np.random.choice([0, 1], size=num_samples, p=[0.75, 0.25])
    email = np.random.choice([0, 1], size=num_samples, p=[0.80, 0.20])
    
    # 3. Formulate correlated Target (approved)
    # High credit score, low DTI, and longer employment increase approval probability
    score_weights = (credit_score - 400) / 450.0 * 0.5
    dti_weights = (1.0 - debt_to_income) * 0.25
    employment_weights = (years_employed / 40.0) * 0.25
    
    approval_prob = score_weights + dti_weights + employment_weights
    approved = (approval_prob > np.random.uniform(0.32, 0.65, size=num_samples)).astype(int)
    
    # 4. Construct DataFrame
    df = pd.DataFrame({
        'gender': gender,
        'owns_car': owns_car,
        'owns_property': owns_property,
        'children_count': children_count,
        'annual_income': annual_income,
        'income_type': income_type,
        'education': education,
        'family_status': family_status,
        'housing_type': housing_type,
        'age': age,
        'years_employed': years_employed,
        'mobile_phone': mobile_phone,
        'work_phone': work_phone,
        'email': email,
        'family_members': family_members,
        'credit_score': credit_score,
        'existing_loans': existing_loans,
        'debt_to_income': debt_to_income,
        'approved': approved
    })
    
    # Insert duplicates and missing values to test preprocessing robustness
    df.loc[np.random.choice(num_samples, size=30), 'annual_income'] = np.nan
    df.loc[np.random.choice(num_samples, size=30), 'credit_score'] = np.nan
    df.loc[np.random.choice(num_samples, size=20), 'income_type'] = np.nan
    
    duplicates = df.sample(n=25, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} synthetic application records at: {output_path}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "creditcard.csv"
    generate_synthetic_data(dataset_path)
