"""
Model Visualization and Analysis Tools
Generates plots and reports for model evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    confusion_matrix
)
from sklearn.calibration import calibration_curve
import os


class ModelVisualizer:
    """
    Create visualizations for model evaluation and comparison.
    """
    
    def __init__(self, trainer, output_dir=None):
        """
        Initialize visualizer with trainer object.
        
        Args:
            trainer: CKDModelTrainer instance with trained models
            output_dir (str): Directory to save plots
        """
        self.trainer = trainer
        
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'visualizations'
            )
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_roc_curves(self, save=True):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.trainer.results.items():
            fpr, tpr, _ = roc_curve(self.trainer.y_test, results['y_pred_proba'])
            roc_auc = results['roc_auc']
            
            plt.plot(fpr, tpr, lw=2, 
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save:
            filepath = os.path.join(self.output_dir, 'roc_curves.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved ROC curves to {filepath}")
        
        plt.close()
    
    def plot_precision_recall_curves(self, save=True):
        """Plot Precision-Recall curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.trainer.results.items():
            precision, recall, _ = precision_recall_curve(
                self.trainer.y_test, 
                results['y_pred_proba']
            )
            
            plt.plot(recall, precision, lw=2, label=f'{model_name}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save:
            filepath = os.path.join(self.output_dir, 'precision_recall_curves.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved Precision-Recall curves to {filepath}")
        
        plt.close()
    
    def plot_confusion_matrices(self, save=True):
        """Plot confusion matrices for all models."""
        n_models = len(self.trainer.results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_models > 1 else [axes]
        
        target_names = self.trainer.label_encoders['target'].classes_
        
        for idx, (model_name, results) in enumerate(self.trainer.results.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar=False,
                       xticklabels=target_names,
                       yticklabels=target_names)
            
            axes[idx].set_title(f'{model_name}', fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'confusion_matrices.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved confusion matrices to {filepath}")
        
        plt.close()
    
    def plot_calibration_curves(self, save=True):
        """Plot calibration curves for models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.trainer.results.items():
            fraction_of_positives, mean_predicted_value = calibration_curve(
                self.trainer.y_test, 
                results['y_pred_proba'],
                n_bins=10,
                strategy='uniform'
            )
            
            plt.plot(mean_predicted_value, fraction_of_positives, 
                    marker='o', lw=2, label=f'{model_name}')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly Calibrated')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Curves (Reliability Diagram)', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save:
            filepath = os.path.join(self.output_dir, 'calibration_curves.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved calibration curves to {filepath}")
        
        plt.close()
    
    def plot_feature_importance(self, model_name=None, top_n=20, save=True):
        """
        Plot feature importance for tree-based models.
        
        Args:
            model_name (str): Name of model to plot. If None, plots all available.
            top_n (int): Number of top features to show
            save (bool): Whether to save the plot
        """
        tree_models = ['Random Forest', 'XGBoost', 'LightGBM']
        
        if model_name:
            models_to_plot = [model_name] if model_name in tree_models else []
        else:
            models_to_plot = [m for m in tree_models if m in self.trainer.models]
        
        if not models_to_plot:
            print("  ℹ️  No tree-based models available for feature importance")
            return
        
        for model_name in models_to_plot:
            model = self.trainer.models[model_name]
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                continue
            
            # Create DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': self.trainer.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Plot
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance_df, x='importance', y='feature', 
                       palette='viridis')
            plt.title(f'Top {top_n} Feature Importance - {model_name}', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            
            if save:
                filename = f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"  ✓ Saved feature importance for {model_name}")
            
            plt.close()
    
    def plot_metric_comparison(self, save=True):
        """Plot bar chart comparing all metrics across models."""
        # Prepare data
        metrics_data = []
        for model_name, results in self.trainer.results.items():
            metrics_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc']
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Melt for grouped bar plot
        df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Score')
        
        # Plot
        plt.figure(figsize=(14, 8))
        sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', palette='Set2')
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.ylim([0, 1.1])
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'metric_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved metric comparison to {filepath}")
        
        plt.close()
    
    def generate_all_plots(self):
        """Generate all available visualizations."""
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        print("\n📊 Creating plots...")
        
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_confusion_matrices()
        self.plot_calibration_curves()
        self.plot_feature_importance()
        self.plot_metric_comparison()
        
        print(f"\n✓ All visualizations saved to: {self.output_dir}")
        
        return self


def visualize_results(trainer):
    """
    Convenience function to generate all visualizations.
    
    Args:
        trainer: Trained CKDModelTrainer instance
    """
    visualizer = ModelVisualizer(trainer)
    visualizer.generate_all_plots()
    return visualizer
