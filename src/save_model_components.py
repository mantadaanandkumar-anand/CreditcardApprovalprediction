# ==============================================================================
# Credit Card Approval Prediction System - Model Saving Module
# ==============================================================================

from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline


def save_model_components(pipeline: Pipeline, output_dir: str = "models") -> None:
    """
    Decomposes a fitted scikit-learn training pipeline into its separate constituent 
    parts (Estimator Model, One-Hot Encoder, and StandardScaler) and serializes 
    them individually to disk using Joblib.
    """
    # 1. Establish the output directory path object
    save_path = Path(output_dir)
    
    # 2. Recursively build the directory on the system if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 3. Extract the trained classifier model step from the pipeline steps
    model = pipeline.named_steps['classifier']
    
    # 4. Extract the ColumnTransformer preprocessor step from the pipeline steps
    preprocessor = pipeline.named_steps['preprocessor']
    
    # 5. Retrieve the StandardScaler object from the numerical column sub-pipeline
    scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
    
    # 6. Retrieve the OneHotEncoder object from the categorical column sub-pipeline
    encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    
    # 7. Serialize the classifier model object to model.pkl
    joblib.dump(model, save_path / "model.pkl")
    print(f"Saved trained estimator model to: {save_path / 'model.pkl'}")
    
    # 8. Serialize the StandardScaler scaling object to scaler.pkl
    joblib.dump(scaler, save_path / "scaler.pkl")
    print(f"Saved numerical scaler parameter weights to: {save_path / 'scaler.pkl'}")
    
    # 9. Serialize the OneHotEncoder category encoder object to encoder.pkl
    joblib.dump(encoder, save_path / "encoder.pkl")
    print(f"Saved categorical encoder weights map to: {save_path / 'encoder.pkl'}")
