"""
Quick Model Training - Faster version for demonstration
Trains models with limited hyperparameter search for speed.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model_training import CKDModelTrainer


def main():
    """Run quick training with minimal CV."""
    print("\n" + "⚡" * 40)
    print("QUICK MODEL TRAINING (Fast Mode)")
    print("⚡" * 40)
    
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'processed',
        'kidney_disease_cleaned.csv'
    )
    
    if not os.path.exists(data_path):
        print("\n❌ Error: Cleaned dataset not found!")
        print("   Please run: python run_preprocessing.py")
        return
    
    print("\n📋 Quick Training Configuration:")
    print("  - Models: LR, RF, XGBoost (without extensive CV)")
    print("  - Faster hyperparameter search")
    print("  - Calibration: Yes")
    
    # Create trainer
    trainer = CKDModelTrainer(data_path, random_state=42)
    
    # Prepare data
    trainer.load_and_prepare_data()
    trainer.split_data(test_size=0.2, stratify=True)
    trainer.scale_features(method='robust')
    
    # Train models WITHOUT extensive CV for speed
    print("\n⚡ Training models (quick mode)...")
    trainer.train_logistic_regression(use_cv=False)
    trainer.train_random_forest(use_cv=False)
    
    if True:  # Try XGBoost if available
        try:
            trainer.train_xgboost(use_cv=False)
        except:
            print("  XGBoost not available")
    
    # Compare models
    trainer.compare_models()
    
    # Quick calibration
    print("\n🔧 Calibrating models...")
    trainer.calibrate_models(method='isotonic', cv=3)
    
    # Save models
    trainer.save_models()
    
    print("\n" + "=" * 80)
    print("✅ QUICK TRAINING COMPLETED!")
    print("=" * 80)
    
    print("\n📊 Results:")
    for model_name, metrics in trainer.results.items():
        print(f"\n{model_name}:")
        print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall:    {metrics['recall']:.4f}")
        print(f"  - F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print("\n💡 For full training with CV, run: python train_models.py")
    print("💡 For nested CV evaluation, run: python src/nested_cv.py")


if __name__ == "__main__":
    main()
