"""
Nested Cross-Validation for Robust Model Evaluation
Implements nested CV for unbiased performance estimation with hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class NestedCVEvaluator:
    """
    Nested Cross-Validation for robust model evaluation.
    
    Outer loop: Unbiased performance estimation
    Inner loop: Hyperparameter tuning
    """
    
    def __init__(self, data_path, random_state=42, n_outer_folds=5, n_inner_folds=3):
        """
        Initialize nested CV evaluator.
        
        Args:
            data_path (str): Path to cleaned dataset
            random_state (int): Random seed
            n_outer_folds (int): Number of outer CV folds for evaluation
            n_inner_folds (int): Number of inner CV folds for tuning
        """
        self.data_path = data_path
        self.random_state = random_state
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.results = {}
        
        np.random.seed(random_state)
    
    def load_and_prepare_data(self):
        """Load and prepare data."""
        print("\n" + "=" * 80)
        print("LOADING DATA FOR NESTED CROSS-VALIDATION")
        print("=" * 80)
        
        # Load data
        df = pd.read_csv(self.data_path)
        print(f"\n✓ Loaded dataset: {df.shape}")
        
        # Prepare features and target
        X = df.drop(['classification', 'id'], axis=1, errors='ignore')
        y = df['classification']
        
        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        
        self.X = X.values
        self.y = y
        self.feature_names = X.columns.tolist()
        
        print(f"✓ Features: {self.X.shape[1]}")
        print(f"✓ Samples: {self.X.shape[0]}")
        
        return self
    
    def nested_cv_logistic_regression(self):
        """Nested CV for Logistic Regression."""
        print("\n" + "=" * 80)
        print("NESTED CV: Logistic Regression")
        print("=" * 80)
        
        # Parameter grid for inner CV
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        # Outer CV
        outer_cv = StratifiedKFold(
            n_splits=self.n_outer_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # Inner CV
        inner_cv = StratifiedKFold(
            n_splits=self.n_inner_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        outer_scores = []
        best_params_list = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(self.X, self.y), 1):
            print(f"\n  Outer Fold {fold_idx}/{self.n_outer_folds}")
            
            # Split data
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # Scale features
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(
                LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced'
                ),
                param_grid,
                cv=inner_cv,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Evaluate best model on outer test set
            best_model = grid_search.best_estimator_
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_pred_proba)
            
            outer_scores.append(score)
            best_params_list.append(grid_search.best_params_)
            
            print(f"    Best params: {grid_search.best_params_}")
            print(f"    Outer fold ROC-AUC: {score:.4f}")
        
        # Store results
        self.results['Logistic Regression'] = {
            'scores': outer_scores,
            'mean_score': np.mean(outer_scores),
            'std_score': np.std(outer_scores),
            'best_params': best_params_list
        }
        
        print(f"\n  ✓ Mean ROC-AUC: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
        
        return self
    
    def nested_cv_random_forest(self, n_iter=15):
        """Nested CV for Random Forest."""
        print("\n" + "=" * 80)
        print("NESTED CV: Random Forest")
        print("=" * 80)
        
        # Parameter distribution for inner CV
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        outer_cv = StratifiedKFold(
            n_splits=self.n_outer_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        inner_cv = StratifiedKFold(
            n_splits=self.n_inner_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        outer_scores = []
        best_params_list = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(self.X, self.y), 1):
            print(f"\n  Outer Fold {fold_idx}/{self.n_outer_folds}")
            
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # Scale features
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Inner CV
            random_search = RandomizedSearchCV(
                RandomForestClassifier(
                    random_state=self.random_state,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                param_dist,
                n_iter=n_iter,
                cv=inner_cv,
                scoring='roc_auc',
                random_state=self.random_state,
                n_jobs=-1
            )
            
            random_search.fit(X_train, y_train)
            
            # Evaluate on outer test set
            best_model = random_search.best_estimator_
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_pred_proba)
            
            outer_scores.append(score)
            best_params_list.append(random_search.best_params_)
            
            print(f"    Best params: {random_search.best_params_}")
            print(f"    Outer fold ROC-AUC: {score:.4f}")
        
        self.results['Random Forest'] = {
            'scores': outer_scores,
            'mean_score': np.mean(outer_scores),
            'std_score': np.std(outer_scores),
            'best_params': best_params_list
        }
        
        print(f"\n  ✓ Mean ROC-AUC: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
        
        return self
    
    def nested_cv_xgboost(self, n_iter=15):
        """Nested CV for XGBoost."""
        if not XGBOOST_AVAILABLE:
            print("\n⚠️  XGBoost not available. Skipping...")
            return self
        
        print("\n" + "=" * 80)
        print("NESTED CV: XGBoost")
        print("=" * 80)
        
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        outer_cv = StratifiedKFold(
            n_splits=self.n_outer_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        inner_cv = StratifiedKFold(
            n_splits=self.n_inner_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        outer_scores = []
        best_params_list = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(self.X, self.y), 1):
            print(f"\n  Outer Fold {fold_idx}/{self.n_outer_folds}")
            
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # Scale features
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Calculate scale_pos_weight
            scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
            
            # Inner CV
            random_search = RandomizedSearchCV(
                xgb.XGBClassifier(
                    random_state=self.random_state,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric='logloss'
                ),
                param_dist,
                n_iter=n_iter,
                cv=inner_cv,
                scoring='roc_auc',
                random_state=self.random_state,
                n_jobs=-1
            )
            
            random_search.fit(X_train, y_train)
            
            # Evaluate on outer test set
            best_model = random_search.best_estimator_
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_pred_proba)
            
            outer_scores.append(score)
            best_params_list.append(random_search.best_params_)
            
            print(f"    Best params: {random_search.best_params_}")
            print(f"    Outer fold ROC-AUC: {score:.4f}")
        
        self.results['XGBoost'] = {
            'scores': outer_scores,
            'mean_score': np.mean(outer_scores),
            'std_score': np.std(outer_scores),
            'best_params': best_params_list
        }
        
        print(f"\n  ✓ Mean ROC-AUC: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
        
        return self
    
    def compare_results(self):
        """Compare nested CV results."""
        print("\n" + "=" * 80)
        print("NESTED CV RESULTS SUMMARY")
        print("=" * 80)
        
        if not self.results:
            print("\n⚠️  No results to compare")
            return None
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Mean ROC-AUC': metrics['mean_score'],
                'Std ROC-AUC': metrics['std_score'],
                'Min Score': min(metrics['scores']),
                'Max Score': max(metrics['scores'])
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Mean ROC-AUC', ascending=False)
        
        print("\n📊 Nested Cross-Validation Results:")
        print(df_comparison.to_string(index=False))
        
        # Statistical significance
        print("\n📈 Individual Fold Scores:")
        for model_name, metrics in self.results.items():
            scores_str = ", ".join([f"{s:.4f}" for s in metrics['scores']])
            print(f"  {model_name}: [{scores_str}]")
        
        best_model = df_comparison.iloc[0]['Model']
        best_score = df_comparison.iloc[0]['Mean ROC-AUC']
        best_std = df_comparison.iloc[0]['Std ROC-AUC']
        
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   Mean ROC-AUC: {best_score:.4f} ± {best_std:.4f}")
        
        return df_comparison
    
    def run_nested_cv(self, models='all'):
        """
        Run nested cross-validation for all specified models.
        
        Args:
            models (str or list): 'all' or list of model names
        """
        print("\n" + "🔬" * 40)
        print("NESTED CROSS-VALIDATION FOR ROBUST MODEL EVALUATION")
        print("🔬" * 40)
        print(f"\nOuter CV folds: {self.n_outer_folds}")
        print(f"Inner CV folds: {self.n_inner_folds}")
        
        self.load_and_prepare_data()
        
        if models == 'all':
            models = ['lr', 'rf', 'xgb']
        
        if 'lr' in models:
            self.nested_cv_logistic_regression()
        
        if 'rf' in models:
            self.nested_cv_random_forest(n_iter=15)
        
        if 'xgb' in models:
            self.nested_cv_xgboost(n_iter=15)
        
        self.compare_results()
        
        print("\n" + "=" * 80)
        print("✅ NESTED CROSS-VALIDATION COMPLETED!")
        print("=" * 80)
        
        return self


def main():
    """Main execution function."""
    import os
    
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'processed',
        'kidney_disease_cleaned.csv'
    )
    
    # Create evaluator
    evaluator = NestedCVEvaluator(
        data_path,
        random_state=42,
        n_outer_folds=5,
        n_inner_folds=3
    )
    
    # Run nested CV
    evaluator.run_nested_cv(models='all')
    
    print("\n💡 Nested CV provides unbiased performance estimates")
    print("   Use these results to select the best model architecture")
    print("   Then retrain on full dataset with best hyperparameters")


if __name__ == "__main__":
    main()
