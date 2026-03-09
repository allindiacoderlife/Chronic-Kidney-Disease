"""
Model Prediction Script
Load trained model and make predictions on new data.
"""

import pickle
import numpy as np
import pandas as pd
import os
import glob


class CKDPredictor:
    """
    Make predictions using trained CKD models.
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize predictor.
        
        Args:
            model_dir (str): Directory containing saved models
        """
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        self.model_dir = model_dir
        self.model = None
        self.preprocessing = None
        self.model_name = None
        self.feature_names = None
        
    def load_model(self, model_type='logistic_regression', use_calibrated=True):
        """
        Load a trained model.
        
        Args:
            model_type (str): 'logistic_regression', 'random_forest', or 'xgboost'
            use_calibrated (bool): Whether to use calibrated version
        """
        # Find latest model file
        pattern = f"{model_type}{'_calibrated' if use_calibrated else ''}_*.pkl"
        model_files = glob.glob(os.path.join(self.model_dir, pattern))
        
        if not model_files:
            raise FileNotFoundError(f"No model found matching pattern: {pattern}")
        
        model_path = sorted(model_files)[-1]  # Get latest
        
        print(f"\n📦 Loading model from: {os.path.basename(model_path)}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        self.model_name = model_type.replace('_', ' ').title()
        if use_calibrated:
            self.model_name += " (Calibrated)"
        
        # Load preprocessing
        preprocessing_files = glob.glob(os.path.join(self.model_dir, 'preprocessing_*.pkl'))
        if preprocessing_files:
            preprocessing_path = sorted(preprocessing_files)[-1]
            with open(preprocessing_path, 'rb') as f:
                self.preprocessing = pickle.load(f)
            print(f"✓ Loaded preprocessing from: {os.path.basename(preprocessing_path)}")
            
            # Extract feature names if available
            if 'feature_names' in self.preprocessing:
                self.feature_names = self.preprocessing['feature_names']
            elif 'label_encoders' in self.preprocessing:
                # Get feature names from encoders (excluding target)
                self.feature_names = [k for k in self.preprocessing['label_encoders'].keys() if k != 'target']
        
        print(f"✓ Model loaded: {self.model_name}")
        
        return self
    
    def preprocess_input(self, data):
        """
        Preprocess input data for prediction.
        
        Args:
            data (pd.DataFrame or dict): Input features
        
        Returns:
            np.array: Preprocessed features
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Remove id and classification if present
        if 'id' in data.columns:
            data = data.drop('id', axis=1)
        if 'classification' in data.columns:
            data = data.drop('classification', axis=1)
        
        # Convert numeric-like strings to numbers
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = pd.to_numeric(data[col], errors='ignore')
                except:
                    pass
        
        # Encode categorical features
        if self.preprocessing and 'label_encoders' in self.preprocessing:
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                if col in self.preprocessing['label_encoders']:
                    le = self.preprocessing['label_encoders'][col]
                    # Handle unknown categories
                    data[col] = data[col].apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    data[col] = le.transform(data[col])
        
        # Convert to numpy
        X = data.values
        
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        if self.preprocessing and 'scaler' in self.preprocessing:
            X = self.preprocessing['scaler'].transform(X)
        
        return X
    
    def predict(self, data, return_proba=True):
        """
        Make predictions.
        
        Args:
            data (pd.DataFrame or dict): Input features
            return_proba (bool): Return probabilities if True, else class labels
        
        Returns:
            predictions and probabilities (if return_proba=True)
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Preprocess
        X = self.preprocess_input(data)
        
        # Predict
        predictions = self.model.predict(X)
        
        if return_proba and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            return predictions, probabilities
        else:
            return predictions
    
    def predict_single(self, patient_data):
        """
        Make prediction for a single patient with detailed output.
        
        Args:
            patient_data (dict): Patient features
        
        Returns:
            dict: Prediction results
        """
        predictions, probabilities = self.predict(patient_data, return_proba=True)
        
        # Get class names
        if self.preprocessing and 'label_encoders' in self.preprocessing:
            target_encoder = self.preprocessing['label_encoders'].get('target')
            if target_encoder:
                class_names = target_encoder.classes_
            else:
                class_names = ['Class 0', 'Class 1']
        else:
            class_names = ['Class 0', 'Class 1']
        
        predicted_class = predictions[0]
        predicted_label = class_names[predicted_class]
        
        results = {
            'prediction': predicted_label,
            'predicted_class': int(predicted_class),
            'probability_ckd': float(probabilities[0][0]),
            'probability_not_ckd': float(probabilities[0][1]),
            'confidence': float(max(probabilities[0])),
            'model': self.model_name
        }
        
        return results
    
    def print_prediction(self, results):
        """Pretty print prediction results."""
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"\n🏥 Model: {results['model']}")
        print(f"\n📊 Prediction: {results['prediction']}")
        print(f"   Confidence: {results['confidence']*100:.1f}%")
        print(f"\n📈 Probabilities:")
        print(f"   CKD:     {results['probability_ckd']*100:.1f}%")
        print(f"   Not CKD: {results['probability_not_ckd']*100:.1f}%")
        print("\n" + "=" * 60)


def example_usage():
    """Demonstrate how to use the predictor."""
    
    print("\n" + "🔮" * 30)
    print("CKD PREDICTION EXAMPLE")
    print("🔮" * 30)
    
    # Initialize predictor
    predictor = CKDPredictor()
    
    # Load model (try logistic regression calibrated version)
    try:
        predictor.load_model(
            model_type='logistic_regression',
            use_calibrated=True
        )
    except FileNotFoundError:
        print("\n❌ No trained model found!")
        print("   Please run training first: python quick_train.py")
        return
    
    # Example patient data (using typical feature values)
    example_patient = {
        'age': 48.0,
        'bp': 80.0,
        'sg': 1.02,
        'al': 1.0,
        'su': 0.0,
        'rbc': 'normal',
        'pc': 'normal',
        'pcc': 'notpresent',
        'ba': 'notpresent',
        'bgr': 121.0,
        'bu': 36.0,
        'sc': 1.2,
        'sod': 138.0,
        'pot': 4.4,
        'hemo': 15.4,
        'pcv': '44',
        'wc': '7800',
        'rc': '5.2',
        'htn': 'yes',
        'dm': 'yes',
        'cad': 'no',
        'appet': 'good',
        'pe': 'no',
        'ane': 'no'
    }
    
    print("\n📋 Patient Data:")
    for key, value in example_patient.items():
        print(f"   {key}: {value}")
    
    # Make prediction
    results = predictor.predict_single(example_patient)
    
    # Print results
    predictor.print_prediction(results)
    
    print("\n💡 Usage Tips:")
    print("   - Provide all 24 features for best accuracy")
    print("   - Use calibrated models for reliable probabilities")
    print("   - Consider clinical context with predictions")
    print("   - Validate with medical professionals")


def batch_predict_from_csv(csv_path, predictor=None):
    """
    Make predictions for multiple patients from CSV.
    
    Args:
        csv_path (str): Path to CSV file with patient data
        predictor (CKDPredictor): Initialized predictor (optional)
    
    Returns:
        pd.DataFrame: Results with predictions
    """
    # Load data
    df = pd.read_csv(csv_path)
    print(f"\n📁 Loaded {len(df)} patients from {csv_path}")
    
    # Initialize predictor if not provided
    if predictor is None:
        predictor = CKDPredictor()
        predictor.load_model('logistic_regression', use_calibrated=True)
    
    # Make predictions
    predictions, probabilities = predictor.predict(df, return_proba=True)
    
    # Add results to dataframe
    df['predicted_class'] = predictions
    df['probability_ckd'] = probabilities[:, 0]
    df['probability_not_ckd'] = probabilities[:, 1]
    df['confidence'] = probabilities.max(axis=1)
    
    # Get class labels
    if predictor.preprocessing and 'label_encoders' in predictor.preprocessing:
        target_encoder = predictor.preprocessing['label_encoders'].get('target')
        if target_encoder:
            df['prediction'] = target_encoder.inverse_transform(predictions)
    
    print(f"✓ Predictions complete!")
    print(f"\n📊 Prediction Summary:")
    print(df['prediction'].value_counts())
    
    return df


if __name__ == "__main__":
    # Run example
    example_usage()
    
    print("\n" + "=" * 60)
    print("📚 Additional Usage Examples:")
    print("=" * 60)
    print("""
# Load different model
predictor = CKDPredictor()
predictor.load_model('random_forest', use_calibrated=True)

# Predict single patient
patient = {...}  # Patient features
results = predictor.predict_single(patient)

# Batch predict from CSV
df_results = batch_predict_from_csv('new_patients.csv')
df_results.to_csv('predictions.csv', index=False)
    """)
