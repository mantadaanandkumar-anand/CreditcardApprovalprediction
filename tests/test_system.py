import unittest
import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Setup paths for modular imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing import clean_data, add_engineered_features, select_features, get_preprocessor
from app import app
import app as flask_app

class TestCreditSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # 1. Setup mock pipeline for end-to-end integration tests
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        
        # Small sample training dataframe to fit the ColumnTransformer pipeline structure
        train_df = pd.DataFrame([
            {
                'gender': 'Male', 'owns_car': 'Yes', 'owns_property': 'Yes', 
                'income_type': 'Working', 'education': 'Higher education', 
                'family_status': 'Married', 'housing_type': 'House / apartment',
                'children_count': 1, 'annual_income': 60000.0, 'age': 35, 
                'years_employed': 5.0, 'work_phone': 1, 'email': 1, 
                'family_members': 3, 'credit_score': 720, 'existing_loans': 1, 
                'debt_to_income': 0.15, 'mobile_phone': 1
            },
            {
                'gender': 'Female', 'owns_car': 'No', 'owns_property': 'No', 
                'income_type': 'Pensioner', 'education': 'Secondary / special education', 
                'family_status': 'Single', 'housing_type': 'Rented apartment',
                'children_count': 0, 'annual_income': 22000.0, 'age': 65, 
                'years_employed': 0.0, 'work_phone': 0, 'email': 0, 
                'family_members': 1, 'credit_score': 550, 'existing_loans': 3, 
                'debt_to_income': 0.45, 'mobile_phone': 1
            }
        ])
        y = pd.Series([1, 0])
        
        # Clean, engineer, and select features
        cleaned = clean_data(train_df)
        engineered = add_engineered_features(cleaned)
        final = select_features(engineered)
        
        numeric_features = final.select_dtypes(exclude=['object', 'string']).columns.tolist()
        categorical_features = final.select_dtypes(include=['object', 'string']).columns.tolist()
        
        preprocessor = get_preprocessor(numeric_features, categorical_features)
        
        # Create and fit a simple test pipeline
        cls.mock_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42))
        ])
        cls.mock_pipeline.fit(final, y)
        
        # Inject mock pipeline directly into the Flask application backend
        flask_app.model_pipeline = cls.mock_pipeline
        
        # Setup Flask Test Client
        app.config['TESTING'] = True
        cls.client = app.test_client()

    # ==========================================================================
    # 1. UNIT TESTS: PREPROCESSING & DATA CLEANING
    # ==========================================================================
    
    def test_duplicate_removal(self):
        """Verify duplicate rows are deleted to clean incoming batches."""
        df = pd.DataFrame([
            {'gender': 'Male', 'age': 30, 'annual_income': 50000.0},
            {'gender': 'Male', 'age': 30, 'annual_income': 50000.0}, # Duplicate record
            {'gender': 'Female', 'age': 40, 'annual_income': 70000.0}
        ])
        cleaned = clean_data(df)
        self.assertEqual(len(cleaned), 2)
        
    def test_column_name_standardization(self):
        """Verify column header keys standardise to lower_snake_case formatting."""
        df = pd.DataFrame([
            {'Gender ': 'Male', 'Annual Income': 50000.0, 'AGE': 30}
        ])
        cleaned = clean_data(df)
        self.assertIn('gender', cleaned.columns)
        self.assertIn('annual_income', cleaned.columns)
        self.assertIn('age', cleaned.columns)
        
    def test_datatype_casting_and_median_imputation(self):
        """Verify flags are rounded and cast to integers, and missing values are median-imputed."""
        df = pd.DataFrame([
            {'age': 35.4, 'children_count': np.nan, 'email': np.nan, 'existing_loans': 2.0},
            {'age': np.nan, 'children_count': 1.0, 'email': 1.0, 'existing_loans': np.nan}
        ])
        cleaned = clean_data(df)
        
        # Verify integers mapping
        self.assertTrue(pd.api.types.is_integer_dtype(cleaned['age']))
        self.assertTrue(pd.api.types.is_integer_dtype(cleaned['children_count']))
        self.assertTrue(pd.api.types.is_integer_dtype(cleaned['email']))
        
        # Verify imputers successfully resolved null entries
        self.assertFalse(cleaned['age'].isnull().any())
        self.assertFalse(cleaned['children_count'].isnull().any())
        self.assertFalse(cleaned['email'].isnull().any())

    def test_outlier_capping(self):
        """Verify continuous values above/below IQR bounds are capped to prevent model bias."""
        df = pd.DataFrame([
            {'annual_income': 10000.0, 'years_employed': 1.0, 'debt_to_income': 0.1},
            {'annual_income': 12000.0, 'years_employed': 2.0, 'debt_to_income': 0.2},
            {'annual_income': 15000.0, 'years_employed': 3.0, 'debt_to_income': 0.15},
            {'annual_income': 14000.0, 'years_employed': 1.5, 'debt_to_income': 0.12},
            {'annual_income': 999999.0, 'years_employed': 90.0, 'debt_to_income': 0.95} # Outlier
        ])
        cleaned = clean_data(df)
        self.assertLess(cleaned['annual_income'].max(), 999999.0)
        self.assertLess(cleaned['years_employed'].max(), 90.0)

    def test_categorical_normalization(self):
        """Verify text features are standardized to stripped Title Case format, resolving string mismatches."""
        df = pd.DataFrame([
            {'gender': ' male ', 'income_type': '  WORKING '},
            {'gender': 'female', 'income_type': 'nan'}
        ])
        cleaned = clean_data(df)
        self.assertEqual(cleaned['gender'].iloc[0], 'Male')
        self.assertEqual(cleaned['income_type'].iloc[0], 'Working')
        # Check string 'nan' formats correctly back to actual float NaN objects for ColumnTransformer
        self.assertTrue(pd.isna(cleaned['income_type'].iloc[1]))

    def test_feature_engineering_calculations(self):
        """Verify features (income ratios, employment fraction, FICO flags) compute accurately."""
        df = pd.DataFrame([
            {'annual_income': 60000.0, 'family_members': 3.0, 'years_employed': 6.0, 'age': 30.0, 'credit_score': 720},
            {'annual_income': 40000.0, 'family_members': 1.0, 'years_employed': 2.0, 'age': 40.0, 'credit_score': 650}
        ])
        engineered = add_engineered_features(df)
        
        # Verify ratio division formulas
        self.assertAlmostEqual(engineered['income_per_family_member'].iloc[0], 20000.0)
        self.assertAlmostEqual(engineered['income_per_family_member'].iloc[1], 40000.0)
        
        self.assertAlmostEqual(engineered['employment_to_age_ratio'].iloc[0], 0.2)
        self.assertAlmostEqual(engineered['employment_to_age_ratio'].iloc[1], 0.05)
        
        self.assertEqual(engineered['high_credit_score'].iloc[0], 1)
        self.assertEqual(engineered['high_credit_score'].iloc[1], 0)

    def test_feature_selection_variance(self):
        """Verify zero-variance constant features (e.g. mobile_phone) are removed."""
        df = pd.DataFrame([
            {'age': 25, 'mobile_phone': 1},
            {'age': 35, 'mobile_phone': 1}
        ])
        selected = select_features(df)
        self.assertNotIn('mobile_phone', selected.columns)
        self.assertIn('age', selected.columns)

    # ==========================================================================
    # 2. INTEGRATION TESTS: FLASK WEB ROUTE CONTROLLERS
    # ==========================================================================
    
    def test_home_page_rendering(self):
        """Verify home page loads secure portal title and layout fields successfully."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Automated Credit Approval System', response.data)
        self.assertIn(b'gender', response.data)
        self.assertIn(b'credit_score', response.data)

    def test_successful_form_prediction_flow(self):
        """Verify rendering of credit evaluation result dashboard given a valid form submission."""
        payload = {
            'gender': 'Male',
            'owns_car': 'Yes',
            'owns_property': 'Yes',
            'children_count': '1',
            'annual_income': '65000',
            'income_type': 'Working',
            'education': 'Higher education',
            'family_status': 'Married',
            'housing_type': 'House / apartment',
            'age': '35',
            'years_employed': '8.5',
            'work_phone': '1',
            'email': '1',
            'family_members': '3',
            'credit_score': '750',
            'existing_loans': '1',
            'debt_to_income': '0.12'
        }
        response = self.client.post('/predict', data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Evaluation Outcome', response.data)
        self.assertIn(b'Profile Audit Log', response.data)
        self.assertIn(b'Approval Probability Score', response.data)
        
    def test_successful_api_prediction_json_flow(self):
        """Verify API prediction response outputs expected JSON payload structures."""
        payload = {
            'gender': 'Female',
            'owns_car': 'No',
            'owns_property': 'Yes',
            'children_count': 0,
            'annual_income': 45000.0,
            'income_type': 'Commercial associate',
            'education': 'Secondary / special education',
            'family_status': 'Single',
            'housing_type': 'House / apartment',
            'age': 28,
            'years_employed': 4.0,
            'work_phone': 0,
            'email': 1,
            'family_members': 1,
            'credit_score': 680,
            'existing_loans': 0,
            'debt_to_income': 0.08
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('approved', data)
        self.assertIn('approval_probability', data)
        self.assertIn('status', data)
        self.assertTrue(isinstance(data['approved'], int))
        self.assertTrue(isinstance(data['approval_probability'], float))

    # ==========================================================================
    # 3. EDGE CASES
    # ==========================================================================
    
    def test_zero_income_and_zero_employment(self):
        """Verify feature engineering handles zero values without throwing division-by-zero errors."""
        payload = {
            'gender': 'Female',
            'owns_car': 'No',
            'owns_property': 'No',
            'children_count': 0,
            'annual_income': 0.0,
            'income_type': 'Pensioner',
            'education': 'Secondary / special education',
            'family_status': 'Single',
            'housing_type': 'House / apartment',
            'age': 70,
            'years_employed': 0.0,
            'work_phone': 0,
            'email': 0,
            'family_members': 0, # Extreme Case: 0 members (should map to 1 to avoid division by zero)
            'credit_score': 400,
            'existing_loans': 0,
            'debt_to_income': 0.0
        }
        
        df = pd.DataFrame([payload])
        cleaned = clean_data(df)
        engineered = add_engineered_features(cleaned)
        
        # division-by-zero protection verification
        self.assertEqual(engineered['income_per_family_member'].iloc[0], 0.0)
        self.assertEqual(engineered['employment_to_age_ratio'].iloc[0], 0.0)
        
        # Test API response for edge case profile
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)

    def test_extreme_boundary_credit_scores(self):
        """Verify robust pipeline execution at the absolute limits of credit scores (380 to 850)."""
        # Minimum Credit Score Profile
        low_credit_payload = {
            'gender': 'Male', 'owns_car': 'No', 'owns_property': 'No',
            'children_count': 0, 'annual_income': 18000.0, 'income_type': 'Working',
            'education': 'Secondary / special education', 'family_status': 'Single',
            'housing_type': 'House / apartment', 'age': 25, 'years_employed': 0.5,
            'work_phone': 0, 'email': 0, 'family_members': 1,
            'credit_score': 380, # Min FICO range
            'existing_loans': 8, 'debt_to_income': 0.85
        }
        
        # Maximum Credit Score Profile
        high_credit_payload = low_credit_payload.copy()
        high_credit_payload['credit_score'] = 850 # Max FICO range
        high_credit_payload['existing_loans'] = 0
        high_credit_payload['debt_to_income'] = 0.01
        
        response_low = self.client.post('/api/predict', json=low_credit_payload)
        response_high = self.client.post('/api/predict', json=high_credit_payload)
        
        self.assertEqual(response_low.status_code, 200)
        self.assertEqual(response_high.status_code, 200)

    # ==========================================================================
    # 4. INVALID INPUT VALIDATION
    # ==========================================================================
    
    def test_missing_required_fields_web_form(self):
        """Verify empty fields in submitted web forms return 400 Bad Request."""
        payload = {
            'gender': 'Male',
            'owns_car': '', # Missing field
            'owns_property': 'Yes',
            'children_count': '1',
            'annual_income': '65000',
            'income_type': 'Working',
            'education': 'Higher education',
            'family_status': 'Married',
            'housing_type': 'House / apartment',
            'age': '', # Missing field
            'years_employed': '8.5',
            'work_phone': '1',
            'email': '1',
            'family_members': '3',
            'credit_score': '750',
            'existing_loans': '1',
            'debt_to_income': '0.12'
        }
        response = self.client.post('/predict', data=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Validation Error", response.data)

    def test_invalid_datatypes_web_form(self):
        """Verify alpha-characters inside numeric forms return 400 Bad Request validation errors."""
        payload = {
            'gender': 'Male',
            'owns_car': 'Yes',
            'owns_property': 'Yes',
            'children_count': 'two', # Invalid datatype
            'annual_income': 'fifty thousand', # Invalid datatype
            'income_type': 'Working',
            'education': 'Higher education',
            'family_status': 'Married',
            'housing_type': 'House / apartment',
            'age': '35',
            'years_employed': '8.5',
            'work_phone': '1',
            'email': '1',
            'family_members': '3',
            'credit_score': '750',
            'existing_loans': '1',
            'debt_to_income': '0.12'
        }
        response = self.client.post('/predict', data=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Validation Error", response.data)

    # ==========================================================================
    # 5. PERFORMANCE & RUNTIME BENCHMARKS
    # ==========================================================================
    
    def test_single_inference_latency(self):
        """Verify single API inference latency remains under 50 milliseconds."""
        payload = {
            'gender': 'Male', 'owns_car': 'Yes', 'owns_property': 'Yes',
            'children_count': 1, 'annual_income': 80000.0, 'income_type': 'Working',
            'education': 'Higher education', 'family_status': 'Married',
            'housing_type': 'House / apartment', 'age': 40, 'years_employed': 12.0,
            'work_phone': 1, 'email': 1, 'family_members': 3,
            'credit_score': 780, 'existing_loans': 0, 'debt_to_income': 0.1
        }
        
        start_time = time.perf_counter()
        response = self.client.post('/api/predict', json=payload)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        self.assertEqual(response.status_code, 200)
        self.assertLess(latency_ms, 50.0, f"Single inference threshold exceeded: {latency_ms:.2f}ms")
        print(f"\n[PERFORMANCE] Single inference latency: {latency_ms:.2f} ms")

    def test_batch_inference_throughput(self):
        """Verify batch throughput runs efficiently below 50ms average per prediction over 100 runs."""
        payload = {
            'gender': 'Female', 'owns_car': 'No', 'owns_property': 'Yes',
            'children_count': 0, 'annual_income': 50000.0, 'income_type': 'Working',
            'education': 'Higher education', 'family_status': 'Single',
            'housing_type': 'House / apartment', 'age': 30, 'years_employed': 4.0,
            'work_phone': 0, 'email': 1, 'family_members': 1,
            'credit_score': 710, 'existing_loans': 1, 'debt_to_income': 0.15
        }
        
        runs = 100
        start_time = time.perf_counter()
        for _ in range(runs):
            response = self.client.post('/api/predict', json=payload)
            self.assertEqual(response.status_code, 200)
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / runs
        print(f"[PERFORMANCE] Batch execution (100 runs) completed in: {total_time_ms:.2f} ms")
        print(f"[PERFORMANCE] Average inference latency: {avg_time_ms:.2f} ms")
        self.assertLess(avg_time_ms, 50.0, f"Batch performance threshold exceeded: {avg_time_ms:.2f}ms")

if __name__ == '__main__':
    unittest.main()
