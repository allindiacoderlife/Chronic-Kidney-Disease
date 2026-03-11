"""
Chronic Kidney Disease Data Preprocessing Script
This script performs comprehensive data cleaning and preprocessing on the CKD dataset.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import os

# For text preprocessing (if needed for categorical columns)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Text preprocessing will be limited.")


class CKDDataPreprocessor:
    """
    A comprehensive data preprocessor for Chronic Kidney Disease dataset.
    """
    
    def __init__(self, data_path):
        """
        Initialize the preprocessor with the dataset path.
        
        Args:
            data_path (str): Path to the kidney_disease.csv file
        """
        self.data_path = data_path
        self.df = None
        self.original_shape = None
        self.cleaned_df = None
        
        # Initialize NLTK components if available
        if NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            except LookupError:
                print("Downloading required NLTK data...")
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('punkt_tab', quiet=True)
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
        
    def load_data(self):
        """Load the dataset and display basic information."""
        print("=" * 80)
        print("LOADING DATASET")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        self.original_shape = self.df.shape
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  - Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"  - Memory usage: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        return self
    
    def explore_data(self):
        """Explore and display dataset information."""
        print("\n" + "=" * 80)
        print("DATA EXPLORATION")
        print("=" * 80)
        
        # Display first few rows
        print("\n📊 First 5 rows:")
        print(self.df.head())
        
        # Display basic info
        print("\n📋 Dataset Information:")
        print(f"  - Total entries: {len(self.df)}")
        print(f"  - Total features: {len(self.df.columns)}")
        print(f"  - Column names: {list(self.df.columns)}")
        
        # Data types
        print("\n📝 Data Types:")
        print(self.df.dtypes)
        
        # Missing values
        print("\n❓ Missing Values:")
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_percent
        })
        print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))
        
        # Statistical summary
        print("\n📈 Statistical Summary (Numerical Features):")
        print(self.df.describe())
        
        # Categorical features
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        print(f"\n🏷️  Categorical Features: {list(categorical_cols)}")
        
        # Check class distribution
        if 'classification' in self.df.columns:
            print("\n🎯 Target Variable Distribution:")
            print(self.df['classification'].value_counts())
        
        return self
    
    def remove_duplicates(self):
        """Remove duplicate rows from the dataset."""
        print("\n" + "=" * 80)
        print("REMOVING DUPLICATES")
        print("=" * 80)
        
        initial_rows = len(self.df)
        duplicates = self.df.duplicated().sum()
        
        if duplicates > 0:
            print(f"\n⚠️  Found {duplicates} duplicate rows")
            self.df = self.df.drop_duplicates()
            print(f"✓ Removed {duplicates} duplicate rows")
            print(f"  - Rows before: {initial_rows}")
            print(f"  - Rows after: {len(self.df)}")
        else:
            print("\n✓ No duplicate rows found!")
        
        return self
    
    def remove_special_characters(self):
        """Remove special characters from string columns."""
        print("\n" + "=" * 80)
        print("REMOVING SPECIAL CHARACTERS")
        print("=" * 80)
        
        string_cols = self.df.select_dtypes(include=['object']).columns
        cleaned_count = 0
        
        for col in string_cols:
            # Store original values for comparison
            original_values = self.df[col].copy()
            
            # Remove special characters, keep only alphanumeric and basic punctuation
            self.df[col] = self.df[col].apply(
                lambda x: re.sub(r'[^\w\s.-]', '', str(x)) if pd.notna(x) else x
            )
            
            # Remove extra whitespace
            self.df[col] = self.df[col].apply(
                lambda x: ' '.join(str(x).split()) if pd.notna(x) else x
            )
            
            # Convert 'nan' strings back to NaN
            self.df[col] = self.df[col].replace('nan', np.nan)
            
            # Count changes
            changes = (original_values != self.df[col]).sum()
            if changes > 0:
                cleaned_count += 1
                print(f"  - Cleaned column '{col}': {changes} values modified")
        
        if cleaned_count > 0:
            print(f"\n✓ Special characters removed from {cleaned_count} columns")
        else:
            print("\n✓ No special characters found in string columns")
        
        return self
    
    def text_preprocessing(self, columns=None):
        """
        Apply tokenization, lemmatization, and stopword removal to text columns.
        Note: This is primarily for text-heavy datasets. CKD dataset has categorical
        values that typically shouldn't be lemmatized, but we'll demonstrate the capability.
        
        Args:
            columns (list): List of column names to process. If None, process all object columns.
        """
        print("\n" + "=" * 80)
        print("TEXT PREPROCESSING (Tokenization, Lemmatization, Stopword Removal)")
        print("=" * 80)
        
        if not NLTK_AVAILABLE:
            print("\n⚠️  NLTK not available. Skipping text preprocessing.")
            return self
        
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns.tolist()
            # Exclude classification column from heavy text processing
            if 'classification' in columns:
                columns.remove('classification')
        
        if not columns:
            print("\n✓ No text columns to process")
            return self
        
        print(f"\n📝 Processing columns: {columns}")
        print("Note: For medical categorical data, we'll apply minimal processing")
        
        for col in columns:
            # Only process non-null values
            non_null_mask = self.df[col].notna()
            
            if non_null_mask.sum() == 0:
                continue
                
            processed_values = []
            
            for value in self.df.loc[non_null_mask, col]:
                # Convert to lowercase
                text = str(value).lower().strip()
                
                # Tokenization
                tokens = word_tokenize(text)
                
                # Remove stopwords and lemmatize
                # (For medical data, we'll be conservative)
                processed_tokens = [
                    self.lemmatizer.lemmatize(token) 
                    for token in tokens 
                    if token.isalnum()  # Keep only alphanumeric tokens
                ]
                
                # Join back (for categorical data, we typically keep the original)
                # But we'll show the processed version
                processed_text = ' '.join(processed_tokens) if processed_tokens else text
                processed_values.append(processed_text)
            
            # Store processed values
            self.df.loc[non_null_mask, col] = processed_values
            print(f"  - Processed '{col}': {len(processed_values)} values")
        
        print("\n✓ Text preprocessing completed")
        
        return self
    
    def handle_missing_data(self, strategy='auto'):
        """
        Handle missing or noisy data using various strategies.
        
        Args:
            strategy (str): Strategy for handling missing data
                - 'auto': Automatically choose strategy based on data type and missing percentage
                - 'drop': Drop rows with missing values
                - 'fill_median': Fill numerical columns with median
                - 'fill_mode': Fill categorical columns with mode
        """
        print("\n" + "=" * 80)
        print("HANDLING MISSING DATA")
        print("=" * 80)
        
        missing_before = self.df.isnull().sum().sum()
        print(f"\n📊 Total missing values before: {missing_before}")
        
        if missing_before == 0:
            print("\n✓ No missing values to handle!")
            return self
        
        # Separate numerical and categorical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        if 'id' in numerical_cols:
            numerical_cols.remove('id')
        
        if strategy == 'auto':
            print("\n🔧 Using AUTO strategy:")
            
            # For numerical columns: fill with median
            for col in numerical_cols:
                missing_count = self.df[col].isnull().sum()
                if missing_count > 0:
                    median_value = self.df[col].median()
                    self.df[col].fillna(median_value, inplace=True)
                    print(f"  - Filled '{col}' ({missing_count} values) with median: {median_value:.2f}")
            
            # For categorical columns: fill with mode or 'unknown'
            for col in categorical_cols:
                missing_count = self.df[col].isnull().sum()
                if missing_count > 0:
                    if len(self.df[col].dropna()) > 0:
                        mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'unknown'
                        self.df[col].fillna(mode_value, inplace=True)
                        print(f"  - Filled '{col}' ({missing_count} values) with mode: '{mode_value}'")
                    else:
                        self.df[col].fillna('unknown', inplace=True)
                        print(f"  - Filled '{col}' ({missing_count} values) with 'unknown'")
        
        elif strategy == 'drop':
            initial_rows = len(self.df)
            self.df = self.df.dropna()
            dropped_rows = initial_rows - len(self.df)
            print(f"\n✓ Dropped {dropped_rows} rows with missing values")
        
        elif strategy == 'fill_median':
            for col in numerical_cols:
                missing_count = self.df[col].isnull().sum()
                if missing_count > 0:
                    median_value = self.df[col].median()
                    self.df[col].fillna(median_value, inplace=True)
                    print(f"  - Filled '{col}' with median: {median_value:.2f}")
        
        elif strategy == 'fill_mode':
            for col in categorical_cols:
                missing_count = self.df[col].isnull().sum()
                if missing_count > 0:
                    mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'unknown'
                    self.df[col].fillna(mode_value, inplace=True)
                    print(f"  - Filled '{col}' with mode: '{mode_value}'")
        
        missing_after = self.df.isnull().sum().sum()
        print(f"\n✓ Missing values handled!")
        print(f"  - Missing values before: {missing_before}")
        print(f"  - Missing values after: {missing_after}")
        
        return self
    
    def handle_noisy_data(self):
        """Detect and handle noisy/outlier data."""
        print("\n" + "=" * 80)
        print("HANDLING NOISY DATA (Outliers)")
        print("=" * 80)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in numerical_cols:
            numerical_cols.remove('id')
        
        outliers_info = {}
        
        for col in numerical_cols:
            # Calculate IQR
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            
            if outliers > 0:
                outliers_info[col] = {
                    'count': outliers,
                    'percentage': (outliers / len(self.df)) * 100,
                    'bounds': (lower_bound, upper_bound)
                }
        
        if outliers_info:
            print("\n⚠️  Outliers detected:")
            for col, info in outliers_info.items():
                print(f"  - {col}: {info['count']} outliers ({info['percentage']:.2f}%)")
                print(f"    Valid range: [{info['bounds'][0]:.2f}, {info['bounds'][1]:.2f}]")
            
            print("\n💡 Note: Outliers detected but not removed (may be clinically significant)")
            print("   Consider domain expertise before removing medical outliers.")
        else:
            print("\n✓ No significant outliers detected using IQR method")
        
        return self
    
    def save_cleaned_data(self, output_path=None):
        """
        Save the cleaned dataset to a new CSV file.
        
        Args:
            output_path (str): Path where cleaned data will be saved
        """
        print("\n" + "=" * 80)
        print("SAVING CLEANED DATASET")
        print("=" * 80)
        
        if output_path is None:
            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(self.data_path), '..', 'processed')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f'kidney_disease_cleaned_{timestamp}.csv')
        
        # Save to CSV
        self.df.to_csv(output_path, index=False)
        
        print(f"\n✓ Cleaned dataset saved successfully!")
        print(f"  - Output path: {output_path}")
        print(f"  - Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"  - File size: {os.path.getsize(output_path) / 1024:.2f} KB")
        
        # Also save a simple version without timestamp for easy access
        simple_path = os.path.join(os.path.dirname(output_path), 'kidney_disease_cleaned.csv')
        self.df.to_csv(simple_path, index=False)
        print(f"  - Also saved as: {simple_path}")
        
        self.cleaned_df = self.df.copy()
        
        return self
    
    def generate_report(self):
        """Generate a preprocessing report."""
        print("\n" + "=" * 80)
        print("PREPROCESSING REPORT")
        print("=" * 80)
        
        print(f"\n📋 Original dataset: {self.original_shape[0]} rows × {self.original_shape[1]} columns")
        print(f"📋 Cleaned dataset: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"\n📊 Changes:")
        print(f"  - Rows removed: {self.original_shape[0] - self.df.shape[0]}")
        print(f"  - Data retention: {(self.df.shape[0] / self.original_shape[0]) * 100:.2f}%")
        
        print(f"\n✅ Final missing values: {self.df.isnull().sum().sum()}")
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 80)
        
        return self
    
    def run_full_pipeline(self, missing_strategy='auto', apply_text_processing=False):
        """
        Run the complete preprocessing pipeline.
        
        Args:
            missing_strategy (str): Strategy for handling missing data
            apply_text_processing (bool): Whether to apply NLTK text processing
        """
        print("\n" + "🔬" * 40)
        print("CHRONIC KIDNEY DISEASE DATA PREPROCESSING PIPELINE")
        print("🔬" * 40)
        
        (self.load_data()
             .explore_data()
             .remove_duplicates()
             .remove_special_characters())
        
        if apply_text_processing:
            self.text_preprocessing()
        
        (self.handle_missing_data(strategy=missing_strategy)
             .handle_noisy_data()
             .save_cleaned_data()
             .generate_report())
        
        return self


def main():
    """Main execution function."""
    # Path to the dataset
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'raw',
        'kidney_disease_dataset.csv'
    )
    
    # Create preprocessor instance
    preprocessor = CKDDataPreprocessor(data_path)
    
    # Run the full pipeline
    # Set apply_text_processing=True if you want to apply NLTK processing
    # (Not recommended for this dataset as it contains categorical medical terms)
    preprocessor.run_full_pipeline(
        missing_strategy='auto',
        apply_text_processing=False
    )
    
    print("\n💡 Tips:")
    print("  - Review the cleaned data before using it for modeling")
    print("  - Check the outliers - they might be clinically significant")
    print("  - Consider feature engineering for better model performance")
    print("  - Encode categorical variables before training ML models")


if __name__ == "__main__":
    main()
