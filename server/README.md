# Chronic Kidney Disease Data Preprocessing

This project provides a comprehensive data preprocessing pipeline for the Chronic Kidney Disease dataset.

## Features

✅ **Data Loading & Exploration**
- Load CSV dataset
- Display dataset statistics and information
- Show missing value analysis
- Display data types and distributions

✅ **Data Cleaning**
- Remove duplicate rows
- Remove special characters from text fields
- Clean extra whitespace

✅ **Text Processing** (Optional)
- Tokenization using NLTK
- Lemmatization
- Stopword removal

✅ **Missing Data Handling**
- Multiple strategies (auto, drop, fill with median/mode)
- Intelligent handling based on data type
- Statistical imputation for numerical features
- Mode imputation for categorical features

✅ **Outlier Detection**
- IQR-based outlier detection
- Detailed outlier reporting
- Conservative approach for medical data

✅ **Data Export**
- Save cleaned dataset with timestamp
- Generate preprocessing report
- Track all changes and transformations

## Installation

1. **Create a virtual environment:**
```powershell
python -m venv venv
```

2. **Activate the virtual environment:**
```powershell
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies:**
```powershell
pip install -r requirements.txt
```

4. **Download NLTK data (if using text processing):**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

## Usage

### Run the complete preprocessing pipeline:

```powershell
python src/data_preprocessing.py
```

### Custom usage in your own script:

```python
from src.data_preprocessing import CKDDataPreprocessor

# Initialize preprocessor
preprocessor = CKDDataPreprocessor('data/raw/kidney_disease_dataset.csv')

# Run full pipeline
preprocessor.run_full_pipeline(
    missing_strategy='auto',  # Options: 'auto', 'drop', 'fill_median', 'fill_mode'
    apply_text_processing=False  # Set True to apply NLTK processing
)

# Or run individual steps
preprocessor.load_data()
preprocessor.explore_data()
preprocessor.remove_duplicates()
preprocessor.remove_special_characters()
preprocessor.handle_missing_data(strategy='auto')
preprocessor.handle_noisy_data()
preprocessor.save_cleaned_data()
preprocessor.generate_report()
```

## Output

The cleaned dataset will be saved in the `data/processed/` directory with two files:
- `kidney_disease_cleaned_YYYYMMDD_HHMMSS.csv` - Timestamped version
- `kidney_disease_cleaned.csv` - Latest cleaned version (for easy access)

## Project Structure

```text
server/
├── api/                            # API related code
├── data/                           # Datasets
│   ├── raw/                        # Original datasets
│   │   └── kidney_disease_dataset.csv      
│   └── processed/                  # Cleaned datasets
│       └── kidney_disease_cleaned.csv  
├── models/                         # Saved model artifacts
├── scripts/                        # Executable scripts
├── src/                            # Core ML pipeline modules
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Dataset Information

The model is trained on the `kidney_disease_dataset.csv` dataset. The Chronic Kidney Disease dataset contains the following features:

**Numerical Features:**
- age, bp (blood pressure), sg (specific gravity), al (albumin), su (sugar)
- bgr (blood glucose random), bu (blood urea), sc (serum creatinine)
- sod (sodium), pot (potassium), hemo (hemoglobin)
- pcv (packed cell volume), wc (white blood cell count), rc (red blood cell count)

**Categorical Features:**
- rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane
- classification (target variable: ckd/notckd)

## Preprocessing Steps

1. **Load Data**: Read CSV and display basic information
2. **Explore Data**: Statistical analysis and missing value detection
3. **Remove Duplicates**: Identify and remove duplicate rows
4. **Clean Special Characters**: Remove unwanted characters from text
5. **Text Processing** (Optional): Tokenize, lemmatize, remove stopwords
6. **Handle Missing Data**: Impute using median (numerical) and mode (categorical)
7. **Detect Outliers**: IQR-based outlier detection (informational only)
8. **Save Results**: Export cleaned dataset with comprehensive report

## Notes

- **Medical Data Consideration**: Outliers are detected but not automatically removed, as they may represent clinically significant cases
- **Text Processing**: NLTK processing is optional and not recommended for this dataset's categorical medical terms
- **Missing Data**: The 'auto' strategy uses median for numerical and mode for categorical features
- **Data Integrity**: Original dataset is never modified; all changes are saved to a new file

## Troubleshooting

### NLTK Download Issues
If you encounter NLTK download issues, run:
```python
import nltk
nltk.download('all')
```

### Missing Packages
If you get import errors, ensure all packages are installed:
```powershell
pip install --upgrade -r requirements.txt
```

### Path Issues
Ensure you run the script from the `server` directory:
```powershell
cd "d:\PROGRAMMING\PYTHON\Chronic Kidney Disease\server"
python src/data_preprocessing.py
```

## Next Steps

After preprocessing:
1. Review the cleaned dataset
2. Perform exploratory data analysis (EDA)
3. Encode categorical variables for ML models
4. Split data into training/testing sets
5. Train classification models
6. Evaluate model performance

## License

This preprocessing pipeline is provided as-is for educational and research purposes.
