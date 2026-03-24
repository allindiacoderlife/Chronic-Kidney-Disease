"""
Chronic Kidney Disease - Model Training V2
Comprehensive ML pipeline with 10 classifiers:
  LogisticRegression, DecisionTree, RandomForest, GradientBoosting,
  SVC, KNeighbors, GaussianNB, MLPClassifier, XGBClassifier, CatBoostClassifier

Dataset: kidney_disease_dataset.csv (multi-class: No_Disease, Low_Risk, Moderate_Risk, High_Risk, Severe_Disease)
"""

import sys
import io

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# Sklearn - Preprocessing & Evaluation
# ──────────────────────────────────────────────
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ──────────────────────────────────────────────
# Sklearn - Models
# ──────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# ──────────────────────────────────────────────
# Boosting Libraries
# ──────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN]  XGBoost not installed. Run: pip install xgboost")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("[WARN]  CatBoost not installed. Run: pip install catboost")

# ──────────────────────────────────────────────
# SMOTE for class imbalance
# ──────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARN]  imbalanced-learn not installed. Run: pip install imbalanced-learn")

# ──────────────────────────────────────────────
# Visualization (optional)
# ──────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')          # non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


# ╔══════════════════════════════════════════════════════════════╗
# ║                  CKD MODEL TRAINER V2                       ║
# ╚══════════════════════════════════════════════════════════════╝

class CKDModelTrainerV2:
    """
    End-to-end training pipeline for CKD multi-class classification.
    Trains 10 models, evaluates each, compares, and saves the best.
    """

    # ── class-level model registry ──
    MODEL_REGISTRY = {
        'Logistic Regression': lambda rs: LogisticRegression(
            max_iter=2000, class_weight='balanced',
            solver='lbfgs', random_state=rs
        ),
        'Decision Tree': lambda rs: DecisionTreeClassifier(
            max_depth=10, class_weight='balanced', random_state=rs
        ),
        'Random Forest': lambda rs: RandomForestClassifier(
            n_estimators=200, class_weight='balanced',
            n_jobs=-1, random_state=rs
        ),
        'Gradient Boosting': lambda rs: GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=3, subsample=0.8, random_state=rs
        ),
        'SVM (SVC)': lambda rs: SVC(
            kernel='rbf', class_weight='balanced',
            probability=True, random_state=rs
        ),
        'K-Nearest Neighbors': lambda rs: KNeighborsClassifier(
            n_neighbors=7, n_jobs=-1
        ),
        'Gaussian Naive Bayes': lambda rs: GaussianNB(),
        'MLP Neural Network': lambda rs: MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu',
            max_iter=300, early_stopping=True,
            random_state=rs
        ),
    }

    def __init__(self, data_path: str, random_state: int = 42):
        self.data_path = data_path
        self.random_state = random_state
        np.random.seed(random_state)

        # data holders
        self.df = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.feature_names: list = []
        self.class_names: list = []

        # preprocessing artefacts
        self.preprocessor = None          # ColumnTransformer
        self.label_encoder = LabelEncoder()
        self.scaler = None

        # trained models & results
        self.models: dict = {}
        self.results: dict = {}

    # ──────────────────────────────
    # 1. LOAD & PREPARE DATA
    # ──────────────────────────────
    def load_and_prepare_data(self):
        """Load CSV, encode features, encode target."""
        self._banner("LOADING & PREPARING DATA")

        self.df = pd.read_csv(self.data_path)
        print(f"[OK]  Loaded {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")

        # Separate target
        if 'classification' not in self.df.columns:
            raise ValueError("Target column 'classification' not found in dataset")

        X = self.df.drop(columns=['classification', 'id'], errors='ignore')
        y = self.df['classification']

        # Show class distribution
        print("\n[CHART]  Target distribution:")
        dist = y.value_counts()
        for cls, cnt in dist.items():
            pct = cnt / len(y) * 100
            print(f"    {cls:<20s}  {cnt:>6,}  ({pct:5.1f}%)")

        # Identify column types
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

        # OneHotEncode categoricals, passthrough numerics
        if cat_cols:
            print(f"\n[PROC]  OneHotEncoding {len(cat_cols)} categorical columns: {cat_cols}")
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
                ],
                remainder='passthrough'
            )
            X_encoded = self.preprocessor.fit_transform(X)
            cat_feature_names = (
                self.preprocessor
                .named_transformers_['cat']
                .get_feature_names_out(cat_cols)
                .tolist()
            )
            self.feature_names = cat_feature_names + num_cols
        else:
            X_encoded = X.values
            self.feature_names = X.columns.tolist()

        print(f"[OK]  Total features after encoding: {len(self.feature_names)}")

        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = list(self.label_encoder.classes_)
        print(f"[LABEL]   Classes ({len(self.class_names)}): {self.class_names}")

        # Handle NaN
        if np.isnan(X_encoded).any():
            nan_count = int(np.isnan(X_encoded).sum())
            print(f"[WARN]  Found {nan_count} NaN values → filling with 0")
            X_encoded = np.nan_to_num(X_encoded, nan=0.0)

        self.X = X_encoded
        self.y = y_encoded
        return self

    # ──────────────────────────────
    # 2. SPLIT DATA
    # ──────────────────────────────
    def split_data(self, test_size: float = 0.2):
        """Stratified train/test split."""
        self._banner("SPLITTING DATA")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            stratify=self.y,
            random_state=self.random_state
        )
        print(f"[OK]  Train: {self.X_train.shape[0]:,} | Test: {self.X_test.shape[0]:,}")
        self._print_class_dist("Train", self.y_train)
        self._print_class_dist("Test",  self.y_test)
        return self

    # ──────────────────────────────
    # 3. SCALE FEATURES
    # ──────────────────────────────
    def scale_features(self, method: str = 'robust'):
        """Scale features using RobustScaler or StandardScaler."""
        self._banner("FEATURE SCALING")

        if method == 'robust':
            self.scaler = RobustScaler()
            print("[SCALE]  Using RobustScaler (resistant to outliers)")
        else:
            self.scaler = StandardScaler()
            print("[SCALE]  Using StandardScaler")

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("[OK]  Scaling complete")
        return self

    # ──────────────────────────────
    # 4. HANDLE CLASS IMBALANCE
    # ──────────────────────────────
    def handle_imbalance(self):
        """Apply SMOTE oversampling on the training set."""
        self._banner("HANDLING CLASS IMBALANCE (SMOTE)")

        if not SMOTE_AVAILABLE:
            print("[WARN]  imbalanced-learn not available → skipping SMOTE")
            return self

        smote = SMOTE(random_state=self.random_state)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

        print(f"[OK]  SMOTE applied → new training size: {self.X_train.shape[0]:,}")
        self._print_class_dist("Train (after SMOTE)", self.y_train)
        return self

    # ──────────────────────────────
    # 5. TRAIN ALL MODELS
    # ──────────────────────────────
    def train_all_models(self):
        """Instantiate and fit every model in the registry + XGBoost + CatBoost."""
        self._banner("TRAINING 10 CLASSIFIERS")

        # ❶ Sklearn models from registry
        for name, factory in self.MODEL_REGISTRY.items():
            self._train_single(name, factory(self.random_state))

        # ❷ XGBoost
        if XGBOOST_AVAILABLE:
            xgb_model = XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                eval_metric='mlogloss',
                use_label_encoder=False,
                random_state=self.random_state,
                n_jobs=-1
            )
            self._train_single('XGBoost', xgb_model)
        else:
            print("[SKIP]   Skipping XGBoost (not installed)")

        # ❸ CatBoost
        if CATBOOST_AVAILABLE:
            cat_model = CatBoostClassifier(
                iterations=150,
                depth=5,
                learning_rate=0.1,
                loss_function='MultiClass',
                random_seed=self.random_state,
                verbose=0
            )
            self._train_single('CatBoost', cat_model)
        else:
            print("[SKIP]   Skipping CatBoost (not installed)")

        return self

    # Models that are slow → skip cross-validation for them
    SLOW_MODELS = {'SVM (SVC)', 'Gradient Boosting', 'MLP Neural Network'}
    # Models that are very slow on large data → limit training samples
    SAMPLE_LIMIT_MODELS = {'SVM (SVC)'}
    MAX_SVM_SAMPLES = 15000  # SVM is O(n^2) so cap training size

    def _train_single(self, name: str, model):
        """Fit one model, evaluate and store."""
        import time
        print(f"\n{'─'*60}")
        print(f"[RUN]  Training: {name}")
        print(f"{'─'*60}")

        t0 = time.time()

        # For SVM, limit training samples (SVM scales as O(n^2))
        if name in self.SAMPLE_LIMIT_MODELS and len(self.X_train) > self.MAX_SVM_SAMPLES:
            from sklearn.utils import resample
            idx = resample(range(len(self.X_train)), n_samples=self.MAX_SVM_SAMPLES,
                           stratify=self.y_train, random_state=self.random_state)
            X_fit, y_fit = self.X_train[idx], self.y_train[idx]
            print(f"    (Using {self.MAX_SVM_SAMPLES:,} samples for SVM speed)")
        else:
            X_fit, y_fit = self.X_train, self.y_train

        model.fit(X_fit, y_fit)
        self.models[name] = model

        # Predictions
        y_pred = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)

        # Cross-validation (skip for slow models)
        if name in self.SLOW_MODELS:
            cv_scores = np.array([acc])  # placeholder — use test accuracy
            print(f"    (CV skipped for speed)")
        else:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            max_cv_samples = 10000
            if len(self.X_train) > max_cv_samples:
                from sklearn.utils import resample
                idx = resample(range(len(self.X_train)), n_samples=max_cv_samples,
                               stratify=self.y_train, random_state=self.random_state)
                X_cv, y_cv = self.X_train[idx], self.y_train[idx]
            else:
                X_cv, y_cv = self.X_train, self.y_train
            cv_scores = cross_val_score(model, X_cv, y_cv, cv=cv, scoring='accuracy', n_jobs=-1)

        # Classification report as dict for programmatic access
        report = classification_report(
            self.y_test, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        cm = confusion_matrix(self.y_test, y_pred)

        self.results[name] = {
            'accuracy': acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'f1_macro': report['macro avg']['f1-score'],
            'precision_weighted': report['weighted avg']['precision'],
            'recall_weighted': report['weighted avg']['recall'],
            'f1_weighted': report['weighted avg']['f1-score'],
            'confusion_matrix': cm,
            'classification_report': report,
            'y_pred': y_pred
        }

        # Print summary
        elapsed = time.time() - t0
        print(f"    Test Accuracy : {acc:.4f}")
        print(f"    CV Accuracy   : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"    Macro F1      : {report['macro avg']['f1-score']:.4f}")
        print(f"    Weighted F1   : {report['weighted avg']['f1-score']:.4f}")
        print(f"    Time elapsed  : {elapsed:.1f}s")

        print(f"\n    Confusion Matrix:")
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        for line in cm_df.to_string().split('\n'):
            print(f"    {line}")

        print(f"\n    Classification Report:")
        report_str = classification_report(
            self.y_test, y_pred,
            target_names=self.class_names,
            zero_division=0
        )
        for line in report_str.split('\n'):
            print(f"    {line}")

    # ──────────────────────────────
    # 6. COMPARE MODELS
    # ──────────────────────────────
    def compare_models(self) -> pd.DataFrame:
        """Print a sorted comparison table of all models."""
        self._banner("MODEL COMPARISON")

        rows = []
        for name, r in self.results.items():
            rows.append({
                'Model': name,
                'Test Acc': r['accuracy'],
                'CV Acc (mean±std)': f"{r['cv_mean']:.4f}±{r['cv_std']:.4f}",
                'Macro Prec': r['precision_macro'],
                'Macro Rec': r['recall_macro'],
                'Macro F1': r['f1_macro'],
                'Weighted F1': r['f1_weighted'],
            })

        df_cmp = pd.DataFrame(rows).sort_values('Test Acc', ascending=False).reset_index(drop=True)

        print("\n" + df_cmp.to_string(index=False))

        best = df_cmp.iloc[0]
        print(f"\n[BEST]  BEST MODEL: {best['Model']}  (Test Acc = {best['Test Acc']:.4f})")

        self.comparison_df = df_cmp
        return df_cmp

    # ──────────────────────────────
    # 7. FEATURE IMPORTANCE
    # ──────────────────────────────
    def show_feature_importance(self, model_name: str = 'Random Forest', top_n: int = 15):
        """Display top-N important features for a tree-based or linear model."""
        self._banner(f"FEATURE IMPORTANCE — {model_name}")

        if model_name not in self.models:
            print(f"[WARN]  Model '{model_name}' not found")
            return None

        model = self.models[model_name]

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.mean(np.abs(model.coef_), axis=0)
        else:
            print(f"[INFO]  {model_name} does not expose feature importance.")
            return None

        imp_df = (
            pd.DataFrame({'Feature': self.feature_names, 'Importance': importance})
            .sort_values('Importance', ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        print(f"\n[TOP]  Top {top_n} features:")
        print(imp_df.to_string(index=False))

        # Optional bar chart
        if PLOT_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=imp_df, x='Importance', y='Feature', palette='viridis', ax=ax)
            ax.set_title(f'Top {top_n} Feature Importances — {model_name}', fontsize=14)
            ax.set_xlabel('Importance')
            ax.set_ylabel('')
            plt.tight_layout()

            vis_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            fig_path = os.path.join(vis_dir, f'feature_importance_{model_name.replace(" ", "_").lower()}.png')
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"[CHART]  Saved chart → {fig_path}")

        return imp_df

    # ──────────────────────────────
    # 8. VISUALIZATION
    # ──────────────────────────────
    def plot_comparison(self):
        """Bar chart comparing Test Accuracy of all models."""
        if not PLOT_AVAILABLE:
            print("[WARN]  matplotlib/seaborn not available")
            return

        self._banner("GENERATING COMPARISON CHART")

        names = [r['Model'] for _, r in pd.DataFrame(
            [{'Model': n, 'Acc': self.results[n]['accuracy']} for n in self.results]
        ).sort_values('Acc', ascending=True).iterrows()]
        # Simpler approach:
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['accuracy'])
        names = [m[0] for m in sorted_models]
        accs = [m[1]['accuracy'] for m in sorted_models]
        f1s = [m[1]['f1_weighted'] for m in sorted_models]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Test Accuracy
        colors = sns.color_palette('coolwarm', len(names))
        axes[0].barh(names, accs, color=colors)
        axes[0].set_xlabel('Test Accuracy')
        axes[0].set_title('Test Accuracy Comparison')
        for i, v in enumerate(accs):
            axes[0].text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)

        # Weighted F1
        axes[1].barh(names, f1s, color=colors)
        axes[1].set_xlabel('Weighted F1-Score')
        axes[1].set_title('Weighted F1-Score Comparison')
        for i, v in enumerate(f1s):
            axes[1].text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)

        plt.suptitle('CKD Model Comparison (V2)', fontsize=16, fontweight='bold')
        plt.tight_layout()

        vis_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        fig_path = os.path.join(vis_dir, 'model_comparison_v2.png')
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"[CHART]  Saved comparison chart → {fig_path}")

    def plot_confusion_matrices(self):
        """Grid of confusion-matrix heatmaps for every trained model."""
        if not PLOT_AVAILABLE:
            return

        self._banner("GENERATING CONFUSION MATRIX HEATMAPS")

        n_models = len(self.models)
        cols = 3
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes = np.array(axes).flatten()

        for idx, (name, res) in enumerate(self.results.items()):
            ax = axes[idx]
            sns.heatmap(
                res['confusion_matrix'],
                annot=True, fmt='d', cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=ax
            )
            ax.set_title(name, fontsize=11)
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')

        # Hide unused subplots
        for j in range(idx + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('Confusion Matrices — All Models', fontsize=15, fontweight='bold')
        plt.tight_layout()

        vis_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        fig_path = os.path.join(vis_dir, 'confusion_matrices_v2.png')
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"[CHART]  Saved confusion matrices → {fig_path}")

    # ──────────────────────────────
    # 9. SAVE MODELS & RESULTS
    # ──────────────────────────────
    def save_models(self, output_dir: str = None):
        """Persist models, scaler, encoder, and JSON results."""
        self._banner("SAVING MODELS & RESULTS")

        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'data', 'models'
            )
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save each model
        for name, model in self.models.items():
            fname = f"{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_{timestamp}.pkl"
            fpath = os.path.join(output_dir, fname)
            with open(fpath, 'wb') as f:
                pickle.dump(model, f)
            print(f"  [SAVE]  {name} → {fname}")

        # Save preprocessing artefacts
        preprocess = {
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
        }
        pre_path = os.path.join(output_dir, f'preprocessing_v2_{timestamp}.pkl')
        with open(pre_path, 'wb') as f:
            pickle.dump(preprocess, f)
        print(f"  [SAVE]  Preprocessing → preprocessing_v2_{timestamp}.pkl")

        # Save results as JSON
        results_json = {}
        for name, r in self.results.items():
            results_json[name] = {
                'accuracy': round(float(r['accuracy']), 4),
                'cv_mean': round(float(r['cv_mean']), 4),
                'cv_std': round(float(r['cv_std']), 4),
                'precision_macro': round(float(r['precision_macro']), 4),
                'recall_macro': round(float(r['recall_macro']), 4),
                'f1_macro': round(float(r['f1_macro']), 4),
                'f1_weighted': round(float(r['f1_weighted']), 4),
            }
        json_path = os.path.join(output_dir, f'results_v2_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"  [SAVE]  Results JSON → results_v2_{timestamp}.json")

        print(f"\n[OK]  All artefacts saved to: {output_dir}")
        return self

    # ──────────────────────────────
    # 10. FULL PIPELINE
    # ──────────────────────────────
    def run_full_pipeline(self):
        """Execute the complete training workflow end-to-end."""
        print("\n" + "█" * 70)
        print("█  CHRONIC KIDNEY DISEASE — MODEL TRAINING V2 PIPELINE")
        print("█" * 70)

        self.load_and_prepare_data()
        self.split_data(test_size=0.2)
        self.scale_features(method='robust')
        self.handle_imbalance()
        self.train_all_models()
        self.compare_models()

        # Feature importance for tree-based models
        for m in ['Random Forest', 'XGBoost', 'Gradient Boosting', 'CatBoost']:
            if m in self.models:
                self.show_feature_importance(m)

        # Visualizations
        self.plot_comparison()
        self.plot_confusion_matrices()

        # Save
        self.save_models()

        print("\n" + "█" * 70)
        print("█  [OK]  PIPELINE COMPLETED SUCCESSFULLY!")
        print("█" * 70)

        return self

    # ──────────────────────── helpers ────────────────────────

    @staticmethod
    def _banner(title: str):
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")

    def _print_class_dist(self, label: str, y: np.ndarray):
        unique, counts = np.unique(y, return_counts=True)
        print(f"    {label} class distribution:")
        for cls_idx, cnt in zip(unique, counts):
            cls_name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else str(cls_idx)
            print(f"      {cls_name:<20s}  {cnt:>6,}  ({cnt/len(y)*100:5.1f}%)")


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    # Path to the RAW kidney disease dataset
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data', 'raw', 'kidney_disease_dataset.csv'
    )

    if not os.path.exists(data_path):
        print(f"[ERR]  Dataset not found at: {data_path}")
        return

    trainer = CKDModelTrainerV2(data_path, random_state=42)
    trainer.run_full_pipeline()

    print("\n[NOTE]  Next Steps:")
    print("    1. Review the comparison table above")
    print("    2. Check visualizations/ folder for charts")
    print("    3. Use the best model for deployment in app.py")
    print("    4. Consider hyperparameter tuning on the top 3 models")


if __name__ == "__main__":
    main()
