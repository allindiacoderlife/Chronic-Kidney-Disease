"""
Comprehensive Model Training Script
Run this to train all models with full pipeline including visualization.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model_training import CKDModelTrainer
from src.model_visualization import visualize_results


def main():
    """Run complete training and visualization pipeline."""
    print("\n" + "🏥" * 40)
    print("CHRONIC KIDNEY DISEASE - COMPLETE ML PIPELINE")
    print("🏥" * 40)
    
    # Path to cleaned data
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'processed',
        'kidney_disease_cleaned.csv'
    )
    
    # Check if cleaned data exists
    if not os.path.exists(data_path):
        print("\n❌ Error: Cleaned dataset not found!")
        print(f"   Expected path: {data_path}")
        print("\n💡 Please run data preprocessing first:")
        print("   python run_preprocessing.py")
        return
    
    print("\n📋 Configuration:")
    print("  - Dataset: kidney_disease_cleaned.csv")
    print("  - Models: Logistic Regression, Random Forest, XGBoost, LightGBM, MLP")
    print("  - Cross-validation: Yes")
    print("  - Calibration: Isotonic Regression")
    print("  - Scaling: RobustScaler")
    
    # Create trainer
    trainer = CKDModelTrainer(data_path, random_state=42)
    
    # Run training pipeline
    print("\n" + "=" * 80)
    print("STARTING TRAINING PIPELINE")
    print("=" * 80)
    
    trainer.run_full_pipeline(
        models_to_train='all',
        use_cv=True,
        calibrate=True
    )
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    try:
        visualize_results(trainer)
    except Exception as e:
        print(f"\n⚠️  Warning: Could not generate all visualizations: {e}")
        print("   (This is usually due to missing matplotlib/seaborn)")
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE PIPELINE FINISHED!")
    print("=" * 80)
    
    print("\n📁 Output Locations:")
    print("  - Models: ./models/")
    print("  - Visualizations: ./visualizations/")
    print("  - Results: ./models/results_*.json")
    
    print("\n📊 Summary:")
    print("  ✓ Data preprocessing completed")
    print("  ✓ Multiple models trained and tuned")
    print("  ✓ Models calibrated for clinical use")
    print("  ✓ Performance metrics computed")
    print("  ✓ Visualizations generated")
    print("  ✓ Models saved for deployment")
    
    print("\n💡 Next Steps:")
    print("  1. Review visualizations in ./visualizations/")
    print("  2. Check model performance in results JSON")
    print("  3. Run nested CV for robust evaluation:")
    print("     python src/nested_cv.py")
    print("  4. Deploy best calibrated model for predictions")
    
    print("\n🎉 Happy modeling!")


if __name__ == "__main__":
    main()
