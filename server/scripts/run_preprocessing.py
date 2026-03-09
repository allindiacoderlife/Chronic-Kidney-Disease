"""
Quick Start Script - Run preprocessing without NLTK
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_preprocessing import CKDDataPreprocessor

def main():
    """Run preprocessing without text processing to avoid NLTK downloads."""
    
    # Path to the dataset
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'raw',
        'kidney_disease.csv'
    )
    
    print("\n🚀 Starting Quick Preprocessing (without NLTK text processing)...\n")
    
    # Create preprocessor instance
    preprocessor = CKDDataPreprocessor(data_path)
    
    # Run pipeline without text processing
    preprocessor.run_full_pipeline(
        missing_strategy='auto',
        apply_text_processing=False  # Skip NLTK to avoid download prompts
    )
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*80)
    print("\nYou can find the cleaned dataset in the 'data/processed' directory.")
    print("\nTo run with full text processing (requires NLTK), use:")
    print("  python src/data_preprocessing.py")

if __name__ == "__main__":
    main()
