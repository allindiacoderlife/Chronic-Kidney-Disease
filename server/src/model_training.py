"""
Chronic Kidney Disease Model Training Pipeline
This script implements multiple ML models with proper evaluation and calibration.
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, brier_score_loss
)

# Baseline Models
from sklearn.linear_model import LogisticRegression

# Tree Ensembles
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

# Neural Networks
try:
    from sklearn.neural_network import MLPClassifier
    MLP_AVAILABLE = True
except ImportError:
    MLP_AVAILABLE = False

# Calibration
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("Warning: Matplotlib/Seaborn not available for plotting")


class CKDModelTrainer:
    """
    Comprehensive model training pipeline for Chronic Kidney Disease prediction.
    """
    
    def __init__(self, data_path, random_state=42):
        """
        Initialize the model trainer.
        
        Args:
            data_path (str): Path to the cleaned dataset
            random_state (int): Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.label_encoders = {}
        self.scaler = None
        self.models = {}
        self.results = {}
        self.calibrated_models = {}
        
        np.random.seed(random_state)
        
    def load_and_prepare_data(self):
        """Load data and prepare features for modeling."""
        print("\n" + "=" * 80)
        print("LOADING AND PREPARING DATA")
        print("=" * 80)
        
        # Load cleaned data
        self.df = pd.read_csv(self.data_path)
        print(f"\n✓ Loaded dataset: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        
        # Separate features and target
        if 'classification' not in self.df.columns:
            raise ValueError("Target column 'classification' not found")
        
        X = self.df.drop(['classification', 'id'], axis=1, errors='ignore')
        y = self.df['classification']
        
        print(f"\n📊 Target distribution:")
        print(y.value_counts())
        print(f"\n  Class balance: {(y.value_counts() / len(y) * 100).round(2).to_dict()}")
        
        # Encode categorical features
        print(f"\n🔧 Encoding categorical features...")
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Try to convert numeric-like strings to numbers first
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    # Try to convert to numeric
                    X[col] = pd.to_numeric(X[col], errors='ignore')
                except:
                    pass
        
        # Re-identify categorical columns after numeric conversion
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            print(f"  - Encoded '{col}': {len(le.classes_)} unique values")
        
        # Encode target variable
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        self.label_encoders['target'] = le_target
        print(f"\n  - Target classes: {le_target.classes_}")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        print(f"\n✓ Total features: {len(self.feature_names)}")
        
        # Convert to numpy arrays
        self.X = X.values
        self.y = y
        
        # Check for NaN values and handle them
        if np.isnan(self.X).any():
            print(f"\n⚠️  Found {np.isnan(self.X).sum()} NaN values - filling with 0")
            self.X = np.nan_to_num(self.X, nan=0.0)
        
        return self
    
    def split_data(self, test_size=0.2, stratify=True):
        """Split data into train and test sets."""
        print("\n" + "=" * 80)
        print("SPLITTING DATA")
        print("=" * 80)
        
        stratify_param = self.y if stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        print(f"\n✓ Data split completed:")
        print(f"  - Training set: {self.X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
        print(f"  - Test set: {self.X_test.shape[0]} samples ({test_size*100:.0f}%)")
        print(f"\n  Training set class distribution:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"    Class {cls}: {cnt} ({cnt/len(self.y_train)*100:.1f}%)")
        
        return self
    
    def scale_features(self, method='robust'):
        """
        Scale numerical features.
        
        Args:
            method (str): 'standard' or 'robust' scaling
        """
        print("\n" + "=" * 80)
        print("FEATURE SCALING")
        print("=" * 80)
        
        if method == 'robust':
            self.scaler = RobustScaler()
            print("\n🔧 Using RobustScaler (better for outliers in medical data)")
        else:
            self.scaler = StandardScaler()
            print("\n🔧 Using StandardScaler")
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("✓ Features scaled successfully")
        
        return self
    
    def train_logistic_regression(self, use_cv=True):
        """Train Logistic Regression with regularization."""
        print("\n" + "=" * 80)
        print("TRAINING: Logistic Regression (Baseline)")
        print("=" * 80)
        
        if use_cv:
            print("\n🔍 Using GridSearchCV for hyperparameter tuning...")
            
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000]
            }
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            grid_search = GridSearchCV(
                LogisticRegression(random_state=self.random_state, class_weight='balanced'),
                param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(self.X_train, self.y_train)
            model = grid_search.best_estimator_
            
            print(f"  ✓ Best parameters: {grid_search.best_params_}")
            print(f"  ✓ Best CV score (ROC-AUC): {grid_search.best_score_:.4f}")
        else:
            model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
            model.fit(self.X_train, self.y_train)
        
        self.models['Logistic Regression'] = model
        self._evaluate_model('Logistic Regression', model)
        
        return self
    
    def train_random_forest(self, use_cv=True):
        """Train Random Forest classifier."""
        print("\n" + "=" * 80)
        print("TRAINING: Random Forest")
        print("=" * 80)
        
        if use_cv:
            print("\n🔍 Using RandomizedSearchCV for hyperparameter tuning...")
            
            param_dist = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', 'balanced_subsample']
            }
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            random_search = RandomizedSearchCV(
                RandomForestClassifier(random_state=self.random_state),
                param_dist,
                n_iter=20,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0
            )
            
            random_search.fit(self.X_train, self.y_train)
            model = random_search.best_estimator_
            
            print(f"  ✓ Best parameters: {random_search.best_params_}")
            print(f"  ✓ Best CV score (ROC-AUC): {random_search.best_score_:.4f}")
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            model.fit(self.X_train, self.y_train)
        
        self.models['Random Forest'] = model
        self._evaluate_model('Random Forest', model)
        
        return self
    
    def train_xgboost(self, use_cv=True):
        """Train XGBoost classifier."""
        if not XGBOOST_AVAILABLE:
            print("\n⚠️  XGBoost not available. Skipping...")
            return self
        
        print("\n" + "=" * 80)
        print("TRAINING: XGBoost")
        print("=" * 80)
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = np.sum(self.y_train == 0) / np.sum(self.y_train == 1)
        
        if use_cv:
            print("\n🔍 Using RandomizedSearchCV for hyperparameter tuning...")
            
            param_dist = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2],
                'min_child_weight': [1, 3, 5]
            }
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            random_search = RandomizedSearchCV(
                xgb.XGBClassifier(
                    random_state=self.random_state,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric='logloss'
                ),
                param_dist,
                n_iter=20,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0
            )
            
            random_search.fit(self.X_train, self.y_train)
            model = random_search.best_estimator_
            
            print(f"  ✓ Best parameters: {random_search.best_params_}")
            print(f"  ✓ Best CV score (ROC-AUC): {random_search.best_score_:.4f}")
        else:
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss'
            )
            model.fit(self.X_train, self.y_train)
        
        self.models['XGBoost'] = model
        self._evaluate_model('XGBoost', model)
        
        return self
    
    def train_lightgbm(self, use_cv=True):
        """Train LightGBM classifier."""
        if not LIGHTGBM_AVAILABLE:
            print("\n⚠️  LightGBM not available. Skipping...")
            return self
        
        print("\n" + "=" * 80)
        print("TRAINING: LightGBM")
        print("=" * 80)
        
        if use_cv:
            print("\n🔍 Using RandomizedSearchCV for hyperparameter tuning...")
            
            param_dist = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [20, 31, 50],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'min_child_samples': [5, 10, 20]
            }
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            random_search = RandomizedSearchCV(
                lgb.LGBMClassifier(
                    random_state=self.random_state,
                    class_weight='balanced',
                    verbose=-1
                ),
                param_dist,
                n_iter=20,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0
            )
            
            random_search.fit(self.X_train, self.y_train)
            model = random_search.best_estimator_
            
            print(f"  ✓ Best parameters: {random_search.best_params_}")
            print(f"  ✓ Best CV score (ROC-AUC): {random_search.best_score_:.4f}")
        else:
            model = lgb.LGBMClassifier(
                n_estimators=200,
                random_state=self.random_state,
                class_weight='balanced',
                verbose=-1
            )
            model.fit(self.X_train, self.y_train)
        
        self.models['LightGBM'] = model
        self._evaluate_model('LightGBM', model)
        
        return self
    
    def train_mlp(self, use_cv=False):
        """Train Multi-Layer Perceptron (Neural Network)."""
        if not MLP_AVAILABLE:
            print("\n⚠️  MLP not available. Skipping...")
            return self
        
        print("\n" + "=" * 80)
        print("TRAINING: Multi-Layer Perceptron (Neural Network)")
        print("=" * 80)
        
        if use_cv:
            print("\n🔍 Using GridSearchCV for hyperparameter tuning...")
            
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            grid_search = GridSearchCV(
                MLPClassifier(
                    random_state=self.random_state,
                    max_iter=1000,
                    early_stopping=True
                ),
                param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(self.X_train, self.y_train)
            model = grid_search.best_estimator_
            
            print(f"  ✓ Best parameters: {grid_search.best_params_}")
            print(f"  ✓ Best CV score (ROC-AUC): {grid_search.best_score_:.4f}")
        else:
            model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                random_state=self.random_state,
                max_iter=1000,
                early_stopping=True
            )
            model.fit(self.X_train, self.y_train)
        
        self.models['MLP'] = model
        self._evaluate_model('MLP', model)
        
        return self
    
    def _evaluate_model(self, model_name, model):
        """Evaluate a trained model."""
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='binary')
        recall = recall_score(self.y_test, y_pred, average='binary')
        f1 = f1_score(self.y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
        
        # Print results
        print(f"\n📊 Test Set Performance:")
        print(f"  - Accuracy:  {accuracy:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall:    {recall:.4f}")
        print(f"  - F1-Score:  {f1:.4f}")
        print(f"  - ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\n  Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"    TN: {cm[0][0]:3d}  FP: {cm[0][1]:3d}")
        print(f"    FN: {cm[1][0]:3d}  TP: {cm[1][1]:3d}")
    
    def calibrate_models(self, method='isotonic', cv=5):
        """
        Calibrate model probabilities for clinical decision making.
        
        Args:
            method (str): 'sigmoid' (Platt scaling) or 'isotonic'
            cv (int): Number of cross-validation folds
        """
        print("\n" + "=" * 80)
        print(f"MODEL CALIBRATION ({method.upper()})")
        print("=" * 80)
        print("\n💡 Calibrating probabilities for clinical decision-making...")
        
        for model_name, model in self.models.items():
            print(f"\n🔧 Calibrating {model_name}...")
            
            # Create calibrated classifier
            calibrated = CalibratedClassifierCV(
                model,
                method=method,
                cv=cv
            )
            
            # Fit on training data
            calibrated.fit(self.X_train, self.y_train)
            
            # Evaluate calibrated model
            y_pred_calib = calibrated.predict(self.X_test)
            y_pred_proba_calib = calibrated.predict_proba(self.X_test)[:, 1]
            
            # Calculate Brier score (lower is better)
            brier_original = brier_score_loss(
                self.y_test,
                self.results[model_name]['y_pred_proba']
            )
            brier_calibrated = brier_score_loss(self.y_test, y_pred_proba_calib)
            
            roc_auc_calib = roc_auc_score(self.y_test, y_pred_proba_calib)
            
            print(f"  ✓ Original Brier Score: {brier_original:.4f}")
            print(f"  ✓ Calibrated Brier Score: {brier_calibrated:.4f}")
            print(f"  ✓ Calibrated ROC-AUC: {roc_auc_calib:.4f}")
            
            # Store calibrated model
            self.calibrated_models[model_name] = {
                'model': calibrated,
                'brier_score': brier_calibrated,
                'roc_auc': roc_auc_calib
            }
        
        print("\n✓ All models calibrated successfully")
        
        return self
    
    def compare_models(self):
        """Compare all trained models."""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('ROC-AUC', ascending=False)
        
        print("\n📊 Model Performance Summary:")
        print(df_comparison.to_string(index=False))
        
        # Find best model
        best_model = df_comparison.iloc[0]['Model']
        best_auc = df_comparison.iloc[0]['ROC-AUC']
        
        print(f"\n🏆 Best Model: {best_model} (ROC-AUC: {best_auc:.4f})")
        
        return df_comparison
    
    def save_models(self, output_dir=None):
        """Save all trained models and results."""
        print("\n" + "=" * 80)
        print("SAVING MODELS")
        print("=" * 80)
        
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(self.data_path),
                '..',
                'models'
            )
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        for model_name, model in self.models.items():
            filename = f"{model_name.replace(' ', '_').lower()}_{timestamp}.pkl"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"  ✓ Saved {model_name} to {filename}")
        
        # Save calibrated models
        for model_name, calib_dict in self.calibrated_models.items():
            filename = f"{model_name.replace(' ', '_').lower()}_calibrated_{timestamp}.pkl"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(calib_dict['model'], f)
            print(f"  ✓ Saved calibrated {model_name} to {filename}")
        
        # Save preprocessing objects
        preprocessing = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        preprocess_file = os.path.join(output_dir, f'preprocessing_{timestamp}.pkl')
        with open(preprocess_file, 'wb') as f:
            pickle.dump(preprocessing, f)
        print(f"  ✓ Saved preprocessing objects")
        
        # Save results as JSON
        results_serializable = {}
        for model_name, metrics in self.results.items():
            results_serializable[model_name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'roc_auc': float(metrics['roc_auc'])
            }
        
        results_file = os.path.join(output_dir, f'results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"  ✓ Saved results to results_{timestamp}.json")
        
        print(f"\n✓ All models saved to: {output_dir}")
        
        return self
    
    def run_full_pipeline(self, models_to_train='all', use_cv=True, calibrate=True):
        """
        Run the complete training pipeline.
        
        Args:
            models_to_train (str or list): 'all' or list of model names
            use_cv (bool): Whether to use cross-validation for hyperparameter tuning
            calibrate (bool): Whether to calibrate model probabilities
        """
        print("\n" + "🧠" * 40)
        print("CHRONIC KIDNEY DISEASE MODEL TRAINING PIPELINE")
        print("🧠" * 40)
        
        # Prepare data
        self.load_and_prepare_data()
        self.split_data(test_size=0.2, stratify=True)
        self.scale_features(method='robust')
        
        # Train models
        if models_to_train == 'all':
            models_to_train = ['lr', 'rf', 'xgb', 'lgb', 'mlp']
        
        if 'lr' in models_to_train:
            self.train_logistic_regression(use_cv=use_cv)
        
        if 'rf' in models_to_train:
            self.train_random_forest(use_cv=use_cv)
        
        if 'xgb' in models_to_train:
            self.train_xgboost(use_cv=use_cv)
        
        if 'lgb' in models_to_train:
            self.train_lightgbm(use_cv=use_cv)
        
        if 'mlp' in models_to_train:
            self.train_mlp(use_cv=False)  # MLP CV is slow
        
        # Compare models
        self.compare_models()
        
        # Calibrate models
        if calibrate and len(self.models) > 0:
            self.calibrate_models(method='isotonic', cv=5)
        
        # Save everything
        self.save_models()
        
        print("\n" + "=" * 80)
        print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return self


def main():
    """Main execution function."""
    # Path to cleaned data
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'processed',
        'kidney_disease_cleaned.csv'
    )
    
    # Create trainer instance
    trainer = CKDModelTrainer(data_path, random_state=42)
    
    # Run full pipeline
    trainer.run_full_pipeline(
        models_to_train='all',  # Train all available models
        use_cv=True,  # Use cross-validation for hyperparameter tuning
        calibrate=True  # Calibrate probabilities for clinical use
    )
    
    print("\n💡 Next Steps:")
    print("  - Review model performance metrics")
    print("  - Analyze feature importance")
    print("  - Validate on external dataset if available")
    print("  - Deploy best calibrated model for clinical use")


if __name__ == "__main__":
    main()
