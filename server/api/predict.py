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
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models')
        self.model_dir = model_dir
        self.model = None
        self.preprocessing = None
        self.model_name = None
        self.feature_names = None
        
    def load_model(self, model_type='logistic_regression', use_calibrated=True):
        """
        Load a trained model.
        
        Args:
            model_type (str): Model type matching the pickle filename prefix
            use_calibrated (bool): Whether to try calibrated version first
        """
        # Find latest model file - try calibrated first, then uncalibrated
        model_files = []
        if use_calibrated:
            pattern = f"{model_type}_calibrated_*.pkl"
            model_files = glob.glob(os.path.join(self.model_dir, pattern))
        
        if not model_files:
            # Fallback to non-calibrated (V2 models are not calibrated)
            pattern = f"{model_type}_*.pkl"
            model_files = glob.glob(os.path.join(self.model_dir, pattern))
            # Exclude preprocessing files
            model_files = [f for f in model_files if 'preprocessing' not in f and 'results' not in f]
        
        if not model_files:
            raise FileNotFoundError(f"No model found matching type: {model_type}")
        
        model_path = sorted(model_files)[-1]  # Get latest
        model_basename = os.path.basename(model_path)
        
        print(f"\n[LOAD] Loading model from: {model_basename}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        self.model_name = model_type.replace('_', ' ').title()
        if 'calibrated' in model_path:
            self.model_name += " (Calibrated)"
        
        # Extract timestamp from model filename to match preprocessing
        # Filenames look like: random_forest_20260324_115420.pkl
        import re
        ts_match = re.search(r'(\d{8}_\d{6})\.pkl$', model_basename)
        model_timestamp = ts_match.group(1) if ts_match else None
        
        # Load matching preprocessing
        self._load_preprocessing(model_timestamp)
        
        print(f"[OK] Model loaded: {self.model_name}")
        
        return self
    
    def _load_preprocessing(self, model_timestamp=None):
        """Load preprocessing artefacts matching the model's training run.
        
        Args:
            model_timestamp: The timestamp string from the model filename (e.g. '20260324_115420').
                           If provided, tries to find preprocessing with matching timestamp first.
        """
        preprocessing_path = None
        
        if model_timestamp:
            # Try exact timestamp match first (V2 then V1)
            v2_exact = os.path.join(self.model_dir, f'preprocessing_v2_{model_timestamp}.pkl')
            v1_exact = os.path.join(self.model_dir, f'preprocessing_{model_timestamp}.pkl')
            
            if os.path.exists(v2_exact):
                preprocessing_path = v2_exact
            elif os.path.exists(v1_exact):
                preprocessing_path = v1_exact
        
        # Fallback: find the latest preprocessing file
        if preprocessing_path is None:
            v2_files = glob.glob(os.path.join(self.model_dir, 'preprocessing_v2_*.pkl'))
            v1_files = glob.glob(os.path.join(self.model_dir, 'preprocessing_*.pkl'))
            v1_files = [f for f in v1_files if '_v2_' not in f]
            
            all_files = v2_files + v1_files
            if all_files:
                preprocessing_path = sorted(all_files)[-1]
        
        if preprocessing_path:
            with open(preprocessing_path, 'rb') as f:
                self.preprocessing = pickle.load(f)
            print(f"[OK] Loaded preprocessing from: {os.path.basename(preprocessing_path)}")
            
            # Extract feature names
            if 'feature_names' in self.preprocessing:
                self.feature_names = self.preprocessing['feature_names']
            elif 'label_encoders' in self.preprocessing:
                self.feature_names = [k for k in self.preprocessing['label_encoders'].keys() if k != 'target']
    
    def preprocess_input(self, data):
        """
        Preprocess input data for prediction.
        Supports both V1 (LabelEncoder per column) and V2 (ColumnTransformer) formats.
        
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
        
        # ── V2 format: ColumnTransformer (OneHotEncoder) ──
        if self.preprocessing and 'preprocessor' in self.preprocessing and self.preprocessing['preprocessor'] is not None:
            print("[DEBUG] Using V2 preprocessor (ColumnTransformer) branch")
            ct = self.preprocessing['preprocessor']
            X = ct.transform(data)
            X = np.nan_to_num(X, nan=0.0)
        
        # ── V1 format: feature_names have OneHot patterns (rbc_normal, htn_yes, etc.) ──
        elif self.feature_names and any('_' in fn for fn in self.feature_names[:30]):
            print("[DEBUG] Using V1 manual one-hot branch. Features:", self.feature_names[:5])
            # Manually reconstruct one-hot encoding from feature names
            X = self._manual_onehot_encode(data)
            X = np.nan_to_num(X, nan=0.0)
        
        else:
            print("[DEBUG] Using fallback raw values branch")
            X = data.values
            X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        if self.preprocessing and 'scaler' in self.preprocessing and self.preprocessing['scaler'] is not None:
            X = self.preprocessing['scaler'].transform(X)
        
        return X
    
    def _manual_onehot_encode(self, data):
        """Manually reconstruct one-hot encoding based on feature_names.
        
        V1 preprocessing has feature_names like:
            ['rbc_abnormal', 'rbc_normal', 'rbc_nan', ..., 'age', 'bp', ...]
        We detect categorical groups (e.g. rbc_*) and create one-hot columns,
        then append numeric columns in the correct order.
        """
        result = np.zeros((len(data), len(self.feature_names)), dtype=float)
        
        # Identify raw categorical and numeric columns from data
        cat_cols = data.select_dtypes(include=['object']).columns.tolist()
        num_cols = data.select_dtypes(exclude=['object']).columns.tolist()
        
        for row_idx in range(len(data)):
            for feat_idx, feat_name in enumerate(self.feature_names):
                # Check if this is a one-hot encoded feature (contains _)
                # Pattern: {col}_{value}  e.g. rbc_normal, htn_yes, pcc_notpresent
                matched = False
                
                for cat_col in cat_cols:
                    if feat_name.startswith(cat_col + '_'):
                        suffix = feat_name[len(cat_col) + 1:]  # e.g. 'normal', 'yes', 'nan'
                        raw_val = str(data.iloc[row_idx][cat_col]).strip().lower()
                        
                        if suffix == 'nan' and (raw_val == '' or raw_val == 'nan'):
                            result[row_idx, feat_idx] = 1.0
                        elif raw_val == suffix.lower():
                            result[row_idx, feat_idx] = 1.0
                        matched = True
                        break
                
                if not matched and feat_name in num_cols:
                    # This is a numeric column
                    try:
                        result[row_idx, feat_idx] = float(data.iloc[row_idx][feat_name])
                    except (ValueError, TypeError):
                        result[row_idx, feat_idx] = 0.0
                elif not matched and feat_name in data.columns:
                    # Try to get numeric value from column
                    try:
                        result[row_idx, feat_idx] = float(data.iloc[row_idx][feat_name])
                    except (ValueError, TypeError):
                        result[row_idx, feat_idx] = 0.0
        
        return result
    
    def _get_class_names(self):
        """Get class names from preprocessing (supports V1 and V2)."""
        if self.preprocessing:
            # V2: class_names stored directly
            if 'class_names' in self.preprocessing:
                return self.preprocessing['class_names']
            # V2: label_encoder (single)
            if 'label_encoder' in self.preprocessing:
                return list(self.preprocessing['label_encoder'].classes_)
            # V1: label_encoders dict with 'target' key
            if 'label_encoders' in self.preprocessing:
                target_encoder = self.preprocessing['label_encoders'].get('target')
                if target_encoder:
                    return list(target_encoder.classes_)
        return [f'Class {i}' for i in range(5)]
    
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
        
        # Get class names - supports V1 and V2 preprocessing formats
        class_names = self._get_class_names()
        
        predicted_class = predictions[0]
        predicted_label = class_names[predicted_class]
        
        prob_dict = {}
        for i, cls_name in enumerate(class_names):
            # Normalizing the class name to be snake_case
            prob_key = str(cls_name).lower().replace(' ', '_')
            if prob_key == 'notckd':
                prob_key = 'not_ckd'
            prob_dict[prob_key] = float(probabilities[0][i])
            
        # Frontend expects 'ckd' to be class 1 and 'notckd' to be class 0
        normalized_class = 1 if str(predicted_label).lower() == 'ckd' else 0
        
        results = {
            'prediction': predicted_label,
            'predicted_class': normalized_class,
            'probabilities': prob_dict,
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
        for cls_name, prob in results['probabilities'].items():
            print(f"   {cls_name.replace('_', ' ').title()}: {prob*100:.1f}%")
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
    df['confidence'] = probabilities.max(axis=1)
    
    # Get class labels for probabilities
    class_names = predictor._get_class_names()
    for i, cls_name in enumerate(class_names):
        prob_key = f"probability_{str(cls_name).lower().replace(' ', '_')}"
        df[prob_key] = probabilities[:, i]
    
    # Get class labels as text
    df['prediction'] = [class_names[p] for p in predictions]
    
    print(f"[OK] Predictions complete!")
    print(f"\nPrediction Summary:")
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
