"""
API Client Example
Demonstrates how to use the CKD Prediction API.
"""

import requests
import json


class CKDAPIClient:
    """Client for interacting with the CKD Prediction API."""
    
    def __init__(self, base_url='http://localhost:5000'):
        """
        Initialize API client.
        
        Args:
            base_url (str): Base URL of the API server
        """
        self.base_url = base_url
    
    def health_check(self):
        """Check if API is healthy."""
        try:
            response = requests.get(f'{self.base_url}/health')
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def predict(self, patient_data):
        """
        Make a prediction for a single patient.
        
        Args:
            patient_data (dict): Patient features
        
        Returns:
            dict: Prediction results
        """
        try:
            response = requests.post(
                f'{self.base_url}/predict',
                json=patient_data,
                headers={'Content-Type': 'application/json'}
            )
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def predict_batch(self, patients):
        """
        Make predictions for multiple patients.
        
        Args:
            patients (list): List of patient data dictionaries
        
        Returns:
            dict: Batch prediction results
        """
        try:
            response = requests.post(
                f'{self.base_url}/predict/batch',
                json={'patients': patients},
                headers={'Content-Type': 'application/json'}
            )
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def get_model_info(self):
        """Get model information."""
        try:
            response = requests.get(f'{self.base_url}/model/info')
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def get_features(self):
        """Get list of required features."""
        try:
            response = requests.get(f'{self.base_url}/features')
            return response.json()
        except Exception as e:
            return {'error': str(e)}


def example_single_prediction():
    """Example: Single patient prediction."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Single Patient Prediction")
    print("=" * 60)
    
    # Initialize client
    client = CKDAPIClient()
    
    # Check health
    health = client.health_check()
    print(f"\n✓ API Health: {health.get('status', 'unknown')}")
    
    # Example patient data
    patient = {
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
    
    # Make prediction
    result = client.predict(patient)
    
    if result.get('success'):
        print(f"\n📊 Prediction: {result['prediction'].upper()}")
        print(f"   Confidence: {result['confidence']*100:.1f}%")
        print(f"\n   Probabilities:")
        print(f"   - CKD:     {result['probabilities']['ckd']*100:.1f}%")
        print(f"   - Not CKD: {result['probabilities']['not_ckd']*100:.1f}%")
        print(f"\n   Model: {result['model']}")
    else:
        print(f"\n❌ Error: {result.get('error')}")


def example_batch_prediction():
    """Example: Batch prediction."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Batch Prediction")
    print("=" * 60)
    
    client = CKDAPIClient()
    
    # Multiple patients
    patients = [
        {
            'age': 48.0, 'bp': 80.0, 'sg': 1.02, 'al': 1.0, 'su': 0.0,
            'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
            'bgr': 121.0, 'bu': 36.0, 'sc': 1.2, 'sod': 138.0, 'pot': 4.4,
            'hemo': 15.4, 'pcv': '44', 'wc': '7800', 'rc': '5.2',
            'htn': 'yes', 'dm': 'yes', 'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no'
        },
        {
            'age': 62.0, 'bp': 90.0, 'sg': 1.015, 'al': 0.0, 'su': 0.0,
            'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
            'bgr': 98.0, 'bu': 22.0, 'sc': 0.9, 'sod': 142.0, 'pot': 4.0,
            'hemo': 14.2, 'pcv': '42', 'wc': '8000', 'rc': '5.0',
            'htn': 'no', 'dm': 'no', 'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no'
        }
    ]
    
    # Make batch prediction
    result = client.predict_batch(patients)
    
    if result.get('success'):
        print(f"\n✓ Total patients: {result['total_patients']}")
        print(f"✓ Successful: {result['successful_predictions']}")
        print(f"✓ Failed: {result['failed_predictions']}")
        
        print("\n📊 Individual Predictions:")
        for pred in result['predictions']:
            if pred.get('success'):
                print(f"\n   Patient {pred['index'] + 1}:")
                print(f"   - Prediction: {pred['prediction']}")
                print(f"   - Confidence: {pred['confidence']*100:.1f}%")
            else:
                print(f"\n   Patient {pred['index'] + 1}: Error - {pred.get('error')}")
    else:
        print(f"\n❌ Error: {result.get('error')}")


def example_model_info():
    """Example: Get model information."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Model Information")
    print("=" * 60)
    
    client = CKDAPIClient()
    
    # Get model info
    info = client.get_model_info()
    
    if info.get('success'):
        print(f"\n📋 Model: {info['model_name']}")
        print(f"   Type: {info['model_type']}")
        print(f"   Features: {info['features_count']}")
        print(f"   Calibrated: {info['calibrated']}")
        
        if info.get('performance'):
            perf = info['performance']
            print(f"\n   Performance:")
            print(f"   - Accuracy:  {perf.get('accuracy', 0)*100:.1f}%")
            print(f"   - Precision: {perf.get('precision', 0)*100:.1f}%")
            print(f"   - Recall:    {perf.get('recall', 0)*100:.1f}%")
            print(f"   - F1-Score:  {perf.get('f1_score', 0):.4f}")
            print(f"   - ROC-AUC:   {perf.get('roc_auc', 0):.4f}")
    else:
        print(f"\n❌ Error: {info.get('error')}")


def example_get_features():
    """Example: Get required features."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Required Features")
    print("=" * 60)
    
    client = CKDAPIClient()
    
    # Get features
    features = client.get_features()
    
    if features.get('success'):
        print(f"\n✓ Total features required: {features['total_features']}")
        print("\n📋 Feature Details:")
        
        for feature in features['features'][:10]:  # Show first 10
            print(f"\n   {feature['name']}:")
            print(f"   - Type: {feature['type']}")
            print(f"   - Description: {feature['description']}")
            if feature.get('unit'):
                print(f"   - Unit: {feature['unit']}")
            if feature.get('values'):
                print(f"   - Values: {', '.join(feature['values'])}")
        
        print("\n   ... (and 14 more features)")
    else:
        print(f"\n❌ Error: {features.get('error')}")


def example_curl_commands():
    """Print example curl commands."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Using CURL")
    print("=" * 60)
    
    print("\n1. Health Check:")
    print("   curl http://localhost:5000/health")
    
    print("\n2. Single Prediction:")
    print("""   curl -X POST http://localhost:5000/predict \\
     -H "Content-Type: application/json" \\
     -d '{"age": 48.0, "bp": 80.0, "sg": 1.02, ...}'""")
    
    print("\n3. Model Info:")
    print("   curl http://localhost:5000/model/info")
    
    print("\n4. Get Features:")
    print("   curl http://localhost:5000/features")


if __name__ == "__main__":
    print("\n" + "🔷" * 30)
    print("CKD PREDICTION API CLIENT EXAMPLES")
    print("🔷" * 30)
    
    print("\n⚠️  Make sure the API server is running:")
    print("   python app.py")
    
    try:
        # Run examples
        example_single_prediction()
        example_batch_prediction()
        example_model_info()
        example_get_features()
        example_curl_commands()
        
        print("\n" + "=" * 60)
        print("✅ ALL EXAMPLES COMPLETED")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n" + "=" * 60)
        print("❌ ERROR: Cannot connect to API server")
        print("=" * 60)
        print("\nPlease start the API server first:")
        print("   python app.py")
        print("\nThen run this script again.")
