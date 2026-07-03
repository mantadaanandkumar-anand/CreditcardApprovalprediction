# ==============================================================================
# Credit Card Approval Prediction System - Flask Web Application Backend
# ==============================================================================

import sys
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify

# Add project root to path for modular imports
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing import clean_data, add_engineered_features, select_features

app = Flask(__name__)

# ------------------------------------------------------------------------------
# Model Loading Configuration
# ------------------------------------------------------------------------------
# Determine the best available model file on startup
MODEL_DIR = project_root / "models"
PIPELINE_PATH = MODEL_DIR / "best_model.joblib"
FALLBACK_PATH = MODEL_DIR / "xgboost_model.joblib"

model_pipeline = None

def load_prediction_model():
    """Loads the trained model pipeline from joblib files."""
    global model_pipeline
    try:
        if PIPELINE_PATH.exists():
            model_pipeline = joblib.load(PIPELINE_PATH)
            print(f"[INFO] Loaded optimal model pipeline from: {PIPELINE_PATH}")
        elif FALLBACK_PATH.exists():
            model_pipeline = joblib.load(FALLBACK_PATH)
            print(f"[INFO] Loaded fallback ensemble pipeline from: {FALLBACK_PATH}")
        else:
            print("[WARNING] No serialized model pipeline found. Runs will fail until models are trained.")
    except Exception as e:
        print(f"[ERROR] Failed to load model pipeline: {str(e)}")

# Trigger model load on application start
load_prediction_model()

# ------------------------------------------------------------------------------
# Web Route Handlers
# ------------------------------------------------------------------------------

@app.route("/")
def home():
    """Renders the applicant input entry form page (integrated with Bootstrap)."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles form submission, validates entries, runs pipeline inference, 
    and renders the result dashboard.
    """
    global model_pipeline
    
    # Reload model on request if it failed to load on startup
    if model_pipeline is None:
        load_prediction_model()
        if model_pipeline is None:
            return jsonify({"error": "Prediction model not found. Please train models first."}), 500
            
    try:
        # Collect raw form inputs
        form_data = request.form.to_dict()
        
        # ----------------------------------------------------------------------
        # Input Validation & Datatype Casting
        # ----------------------------------------------------------------------
        # Field mapping for validation
        int_fields = ['children_count', 'age', 'family_members', 'credit_score', 'existing_loans', 'work_phone', 'email']
        float_fields = ['annual_income', 'years_employed', 'debt_to_income']
        categorical_fields = ['gender', 'owns_car', 'owns_property', 'income_type', 'education', 'family_status', 'housing_type']
        
        validated_data = {}
        
        # Parse and validate integer fields
        for field in int_fields:
            val = form_data.get(field, '').strip()
            if not val:
                return f"Validation Error: Field '{field}' is required.", 400
            try:
                validated_data[field] = int(val)
            except ValueError:
                return f"Validation Error: Field '{field}' must be an integer.", 400
                
        # Parse and validate float fields
        for field in float_fields:
            val = form_data.get(field, '').strip()
            if not val:
                return f"Validation Error: Field '{field}' is required.", 400
            try:
                validated_data[field] = float(val)
            except ValueError:
                return f"Validation Error: Field '{field}' must be a numeric value.", 400

        # Validate categorical fields
        for field in categorical_fields:
            val = form_data.get(field, '').strip()
            if not val:
                return f"Validation Error: Field '{field}' is required.", 400
            validated_data[field] = val

        # Set constant 'mobile_phone' (dropped during feature selection but expected in raw preprocessing schema)
        validated_data['mobile_phone'] = 1

        # ----------------------------------------------------------------------
        # Inference Step
        # ----------------------------------------------------------------------
        # 1. Convert single application dictionary into a DataFrame row
        app_df = pd.DataFrame([validated_data])
        
        # 2. Apply the identical cleaning steps (snake_case conversion, category Title casing, datatype checks)
        cleaned_app = clean_data(app_df)
        
        # 3. Apply custom feature engineering (engineered ratios)
        engineered_app = add_engineered_features(cleaned_app)
        
        # 4. Apply feature selection (dropping constant mobile_phone columns)
        final_app = select_features(engineered_app)
        
        # 5. Execute model pipeline prediction (applies fit imputation, scaling, encoding, and scores model)
        approval_prob = float(model_pipeline.predict_proba(final_app)[0, 1])
        approval_decision = int(model_pipeline.predict(final_app)[0])
        
        # Determine textual label and bootstrap visual indicator class
        result_label = "Approved" if approval_decision == 1 else "Rejected"
        badge_class = "success" if approval_decision == 1 else "danger"
        
        # Return results rendered inside the result template
        return render_template(
            "result.html",
            prediction=result_label,
            probability=f"{approval_prob:.1%}",
            probability_raw=approval_prob,
            badge_class=badge_class,
            applicant=validated_data
        )

    except Exception as e:
        # Graceful error handler returning debug print logs
        print(f"[SERVER ERROR] Inference failed: {str(e)}")
        return f"System Inference Error: {str(e)}", 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for credit prediction (useful for external services)."""
    global model_pipeline
    if model_pipeline is None:
        return jsonify({"error": "Prediction model not found."}), 500
        
    try:
        json_data = request.get_json(force=True)
        # Convert to DataFrame
        app_df = pd.DataFrame([json_data])
        
        # Apply preprocessing
        cleaned_app = clean_data(app_df)
        engineered_app = add_engineered_features(cleaned_app)
        final_app = select_features(engineered_app)
        
        # Predict
        approval_prob = float(model_pipeline.predict_proba(final_app)[0, 1])
        approval_decision = int(model_pipeline.predict(final_app)[0])
        
        return jsonify({
            "approved": approval_decision,
            "approval_probability": approval_prob,
            "status": "Approved" if approval_decision == 1 else "Rejected"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Start local development server on port 5000
    app.run(host="127.0.0.1", port=5000, debug=True)
