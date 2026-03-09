"""
Flask API for Chronic Kidney Disease Prediction
RESTful API for model deployment and predictions.
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime
import logging

# Import predictor
from api.predict import CKDPredictor

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictors - one for each model type
predictors = {}
current_model = 'logistic_regression'

# Available models configuration
AVAILABLE_MODELS = {
    'logistic_regression': {
        'name': 'Logistic Regression',
        'type': 'logistic_regression',
        'description': 'Linear model with probability calibration'
    }
}


def initialize_predictor(model_type='logistic_regression', use_calibrated=True):
    """
    Initialize a predictor with specified model.
    
    Args:
        model_type: Type of model to load
        use_calibrated: Whether to use calibrated version
    
    Returns:
        CKDPredictor instance or None
    """
    try:
        predictor = CKDPredictor(model_dir='models')
        predictor.load_model(
            model_type=model_type,
            use_calibrated=use_calibrated
        )
        logger.info(f"✓ Model loaded successfully: {predictor.model_name}")
        return predictor
    except Exception as e:
        logger.error(f"❌ Failed to load model {model_type}: {e}")
        return None


def initialize_all_models():
    """Initialize all available models."""
    global predictors, current_model
    
    logger.info("Initializing models...")
    
    # FOR RENDER FREE TIER: Only load the default model to save memory
    default_model = 'logistic_regression'
    model_config = AVAILABLE_MODELS[default_model]
    
    predictor = initialize_predictor(model_config['type'], use_calibrated=True)
    if predictor:
        predictors[default_model] = predictor
        logger.info(f"✓ {model_config['name']} ready")
    else:
        logger.warning(f"⚠ {model_config['name']} failed to load")
    
    # Set default model
    if 'logistic_regression' in predictors:
        current_model = 'logistic_regression'
    elif predictors:
        current_model = list(predictors.keys())[0]
    
    logger.info(f"✓ Initialized {len(predictors)} models (limited for memory)")
    logger.info(f"✓ Current model: {current_model}")


def get_current_predictor():
    """Get the currently active predictor."""
    return predictors.get(current_model)


# Initialize models on startup
initialize_all_models()


# HTML template for home page
HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CKD Prediction API</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        h2 {
            color: #764ba2;
            margin-top: 30px;
        }
        .endpoint {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        .method {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-weight: bold;
            color: white;
            font-size: 12px;
            margin-right: 10px;
        }
        .get { background: #28a745; }
        .post { background: #007bff; }
        code {
            background: #272822;
            color: #f8f8f2;
            padding: 15px;
            display: block;
            border-radius: 5px;
            overflow-x: auto;
            margin: 10px 0;
            font-size: 14px;
        }
        .status {
            padding: 10px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .status.ok {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .feature-list {
            columns: 2;
            -webkit-columns: 2;
            -moz-columns: 2;
        }
        .feature-list li {
            margin: 5px 0;
        }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            background: #667eea;
            color: white;
            border-radius: 3px;
            font-size: 12px;
            margin: 0 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Chronic Kidney Disease Prediction API</h1>
        
        <div class="status {{ status_class }}">
            <strong>Status:</strong> {{ status_message }}
        </div>
        
        <p>Welcome to the CKD Prediction API. This API provides endpoints for predicting chronic kidney disease using machine learning models.</p>
        
        <h2>📊 Current Model</h2>
        <p>
            <strong>Model:</strong> {{ model_name }}<br>
            <strong>Performance:</strong> <span class="badge">Accuracy: 100%</span> <span class="badge">ROC-AUC: 1.0</span><br>
            <strong>Calibration:</strong> <span class="badge">Brier Score: 0.0042</span>
        </p>
        
        <h2>🔗 API Endpoints</h2>
        
        <div class="endpoint">
            <span class="method get">GET</span> <strong>/</strong>
            <p>This page - API documentation and status.</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <strong>/health</strong>
            <p>Health check endpoint. Returns API status and model information.</p>
            <code>curl http://localhost:5000/health</code>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <strong>/predict</strong>
            <p>Make a prediction for a single patient.</p>
            <p><strong>Request Body (JSON):</strong></p>
            <code>{
  "age": 48.0,
  "bp": 80.0,
  "sg": 1.02,
  "al": 1.0,
  "su": 0.0,
  "rbc": "normal",
  "pc": "normal",
  "pcc": "notpresent",
  "ba": "notpresent",
  "bgr": 121.0,
  "bu": 36.0,
  "sc": 1.2,
  "sod": 138.0,
  "pot": 4.4,
  "hemo": 15.4,
  "pcv": "44",
  "wc": "7800",
  "rc": "5.2",
  "htn": "yes",
  "dm": "yes",
  "cad": "no",
  "appet": "good",
  "pe": "no",
  "ane": "no"
}</code>
            <p><strong>Example with curl:</strong></p>
            <code>curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 48.0, "bp": 80.0, ...}'</code>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <strong>/predict/batch</strong>
            <p>Make predictions for multiple patients.</p>
            <p><strong>Request Body (JSON):</strong></p>
            <code>{
  "patients": [
    {"age": 48.0, "bp": 80.0, ...},
    {"age": 62.0, "bp": 90.0, ...}
  ]
}</code>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <strong>/model/info</strong>
            <p>Get detailed information about the loaded model.</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <strong>/features</strong>
            <p>Get list of required features and their descriptions.</p>
        </div>
        
        <h2>📋 Required Features</h2>
        <p>The following 24 features are required for prediction:</p>
        <ul class="feature-list">
            <li><strong>age</strong> - Age in years</li>
            <li><strong>bp</strong> - Blood Pressure (mm/Hg)</li>
            <li><strong>sg</strong> - Specific Gravity</li>
            <li><strong>al</strong> - Albumin</li>
            <li><strong>su</strong> - Sugar</li>
            <li><strong>rbc</strong> - Red Blood Cells (normal/abnormal)</li>
            <li><strong>pc</strong> - Pus Cell (normal/abnormal)</li>
            <li><strong>pcc</strong> - Pus Cell Clumps (present/notpresent)</li>
            <li><strong>ba</strong> - Bacteria (present/notpresent)</li>
            <li><strong>bgr</strong> - Blood Glucose Random (mg/dL)</li>
            <li><strong>bu</strong> - Blood Urea (mg/dL)</li>
            <li><strong>sc</strong> - Serum Creatinine (mg/dL)</li>
            <li><strong>sod</strong> - Sodium (mEq/L)</li>
            <li><strong>pot</strong> - Potassium (mEq/L)</li>
            <li><strong>hemo</strong> - Hemoglobin (g/dL)</li>
            <li><strong>pcv</strong> - Packed Cell Volume (%)</li>
            <li><strong>wc</strong> - White Blood Cell Count</li>
            <li><strong>rc</strong> - Red Blood Cell Count</li>
            <li><strong>htn</strong> - Hypertension (yes/no)</li>
            <li><strong>dm</strong> - Diabetes Mellitus (yes/no)</li>
            <li><strong>cad</strong> - Coronary Artery Disease (yes/no)</li>
            <li><strong>appet</strong> - Appetite (good/poor)</li>
            <li><strong>pe</strong> - Pedal Edema (yes/no)</li>
            <li><strong>ane</strong> - Anemia (yes/no)</li>
        </ul>
        
        <h2>💡 Example Response</h2>
        <code>{
  "success": true,
  "prediction": "ckd",
  "predicted_class": 0,
  "probabilities": {
    "ckd": 0.897,
    "not_ckd": 0.103
  },
  "confidence": 0.897,
  "model": "Logistic Regression (Calibrated)",
  "timestamp": "2025-11-01T12:34:56"
}</code>
        
        <h2>🔒 Error Handling</h2>
        <p>All endpoints return standardized error responses:</p>
        <code>{
  "success": false,
  "error": "Error message",
  "details": "Detailed error information"
}</code>
        
        <h2>📞 Support</h2>
        <p>For issues or questions, please refer to the project documentation.</p>
        
        <p style="margin-top: 40px; text-align: center; color: #666;">
            <small>CKD Prediction API v1.0 | Built with Flask | © 2025</small>
        </p>
    </div>
</body>
</html>
"""


@app.route('/')
def home():
    """Home page with API documentation."""
    predictor = get_current_predictor()
    status_class = "ok" if predictor else "error"
    status_message = f"✓ API is running with {len(predictors)}/{len(AVAILABLE_MODELS)} models loaded" if predictors else "❌ No models loaded"
    model_name = predictor.model_name if predictor else "Not loaded"
    
    return render_template_string(
        HOME_TEMPLATE,
        status_class=status_class,
        status_message=status_message,
        model_name=model_name
    )


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    predictor = get_current_predictor()
    if predictor:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'current_model': current_model,
            'model_name': predictor.model_name,
            'total_models': len(predictors),
            'available_models': len(AVAILABLE_MODELS),
            'timestamp': datetime.now().isoformat()
        }), 200
    else:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'error': 'No models loaded',
            'timestamp': datetime.now().isoformat()
        }), 503


@app.route('/predict', methods=['POST'])
def predict():
    """Make a single prediction."""
    try:
        # Get current predictor
        predictor = get_current_predictor()
        
        # Check if model is loaded
        if not predictor:
            return jsonify({
                'success': False,
                'error': 'Model not loaded',
                'details': 'Please check server logs'
            }), 503
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided',
                'details': 'Request body must contain JSON data'
            }), 400
        
        # Make prediction
        results = predictor.predict_single(data)
        
        # Format response
        response = {
            'success': True,
            'prediction': results['prediction'],
            'predicted_class': results['predicted_class'],
            'probabilities': {
                'ckd': round(results['probability_ckd'], 4),
                'not_ckd': round(results['probability_not_ckd'], 4)
            },
            'confidence': round(results['confidence'], 4),
            'model': results['model'],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction made: {results['prediction']} (confidence: {results['confidence']:.2%})")
        
        return jsonify(response), 200
        
    except KeyError as e:
        return jsonify({
            'success': False,
            'error': 'Missing required feature',
            'details': f'Feature not found: {str(e)}'
        }), 400
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Prediction failed',
            'details': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Make predictions for multiple patients."""
    try:
        # Get current predictor
        predictor = get_current_predictor()
        
        # Check if model is loaded
        if not predictor:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 503
        
        # Get JSON data
        data = request.get_json()
        
        if not data or 'patients' not in data:
            return jsonify({
                'success': False,
                'error': 'Invalid request format',
                'details': 'Request must contain "patients" array'
            }), 400
        
        patients = data['patients']
        
        if not isinstance(patients, list):
            return jsonify({
                'success': False,
                'error': 'Invalid data type',
                'details': '"patients" must be an array'
            }), 400
        
        # Make predictions for all patients
        predictions = []
        for i, patient in enumerate(patients):
            try:
                result = predictor.predict_single(patient)
                predictions.append({
                    'index': i,
                    'success': True,
                    'prediction': result['prediction'],
                    'probabilities': {
                        'ckd': round(result['probability_ckd'], 4),
                        'not_ckd': round(result['probability_not_ckd'], 4)
                    },
                    'confidence': round(result['confidence'], 4)
                })
            except Exception as e:
                predictions.append({
                    'index': i,
                    'success': False,
                    'error': str(e)
                })
        
        # Calculate summary
        successful = sum(1 for p in predictions if p.get('success', False))
        
        response = {
            'success': True,
            'total_patients': len(patients),
            'successful_predictions': successful,
            'failed_predictions': len(patients) - successful,
            'predictions': predictions,
            'model': predictor.model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch prediction: {successful}/{len(patients)} successful")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Batch prediction failed',
            'details': str(e)
        }), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information."""
    predictor = get_current_predictor()
    if not predictor:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 503
    
    # Load results if available
    results_files = glob.glob(os.path.join('models', 'results_*.json'))
    performance = None
    
    if results_files:
        import json
        results_path = sorted(results_files)[-1]
        with open(results_path, 'r') as f:
            all_results = json.load(f)
            # Get results for current model
            model_key = AVAILABLE_MODELS[current_model]['name']
            if model_key in all_results:
                performance = all_results[model_key]
    
    info = {
        'success': True,
        'current_model': current_model,
        'model_name': predictor.model_name,
        'model_type': AVAILABLE_MODELS[current_model]['type'],
        'description': AVAILABLE_MODELS[current_model]['description'],
        'features_count': len(predictor.feature_names) if predictor.feature_names else 24,
        'feature_names': predictor.feature_names if predictor.feature_names else [],
        'calibrated': 'Calibrated' in predictor.model_name,
        'performance': performance,
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(info), 200


@app.route('/features', methods=['GET'])
def get_features():
    """Get list of required features."""
    features = {
        'success': True,
        'total_features': 24,
        'features': [
            {'name': 'age', 'type': 'numeric', 'description': 'Age in years', 'unit': 'years'},
            {'name': 'bp', 'type': 'numeric', 'description': 'Blood Pressure', 'unit': 'mm/Hg'},
            {'name': 'sg', 'type': 'numeric', 'description': 'Specific Gravity', 'unit': ''},
            {'name': 'al', 'type': 'numeric', 'description': 'Albumin', 'unit': ''},
            {'name': 'su', 'type': 'numeric', 'description': 'Sugar', 'unit': ''},
            {'name': 'rbc', 'type': 'categorical', 'description': 'Red Blood Cells', 'values': ['normal', 'abnormal']},
            {'name': 'pc', 'type': 'categorical', 'description': 'Pus Cell', 'values': ['normal', 'abnormal']},
            {'name': 'pcc', 'type': 'categorical', 'description': 'Pus Cell Clumps', 'values': ['present', 'notpresent']},
            {'name': 'ba', 'type': 'categorical', 'description': 'Bacteria', 'values': ['present', 'notpresent']},
            {'name': 'bgr', 'type': 'numeric', 'description': 'Blood Glucose Random', 'unit': 'mg/dL'},
            {'name': 'bu', 'type': 'numeric', 'description': 'Blood Urea', 'unit': 'mg/dL'},
            {'name': 'sc', 'type': 'numeric', 'description': 'Serum Creatinine', 'unit': 'mg/dL'},
            {'name': 'sod', 'type': 'numeric', 'description': 'Sodium', 'unit': 'mEq/L'},
            {'name': 'pot', 'type': 'numeric', 'description': 'Potassium', 'unit': 'mEq/L'},
            {'name': 'hemo', 'type': 'numeric', 'description': 'Hemoglobin', 'unit': 'g/dL'},
            {'name': 'pcv', 'type': 'numeric', 'description': 'Packed Cell Volume', 'unit': '%'},
            {'name': 'wc', 'type': 'numeric', 'description': 'White Blood Cell Count', 'unit': 'cells/cumm'},
            {'name': 'rc', 'type': 'numeric', 'description': 'Red Blood Cell Count', 'unit': 'millions/cumm'},
            {'name': 'htn', 'type': 'categorical', 'description': 'Hypertension', 'values': ['yes', 'no']},
            {'name': 'dm', 'type': 'categorical', 'description': 'Diabetes Mellitus', 'values': ['yes', 'no']},
            {'name': 'cad', 'type': 'categorical', 'description': 'Coronary Artery Disease', 'values': ['yes', 'no']},
            {'name': 'appet', 'type': 'categorical', 'description': 'Appetite', 'values': ['good', 'poor']},
            {'name': 'pe', 'type': 'categorical', 'description': 'Pedal Edema', 'values': ['yes', 'no']},
            {'name': 'ane', 'type': 'categorical', 'description': 'Anemia', 'values': ['yes', 'no']}
        ]
    }
    
    return jsonify(features), 200


@app.route('/models', methods=['GET'])
def list_models():
    """Get list of all available models."""
    models_list = []
    
    for model_key, model_config in AVAILABLE_MODELS.items():
        predictor = predictors.get(model_key)
        
        model_info = {
            'id': model_key,
            'name': model_config['name'],
            'description': model_config['description'],
            'loaded': predictor is not None,
            'active': model_key == current_model
        }
        
        # Add performance metrics if available
        results_files = glob.glob(os.path.join('models', 'results_*.json'))
        if results_files and predictor:
            import json
            results_path = sorted(results_files)[-1]
            with open(results_path, 'r') as f:
                all_results = json.load(f)
                if model_config['name'] in all_results:
                    model_info['performance'] = all_results[model_config['name']]
        
        models_list.append(model_info)
    
    return jsonify({
        'success': True,
        'total_models': len(AVAILABLE_MODELS),
        'loaded_models': len(predictors),
        'current_model': current_model,
        'models': models_list
    }), 200


@app.route('/models/<model_id>', methods=['POST'])
def switch_model(model_id):
    """Switch to a different model."""
    global current_model
    
    if model_id not in AVAILABLE_MODELS:
        return jsonify({
            'success': False,
            'error': 'Invalid model ID',
            'available_models': list(AVAILABLE_MODELS.keys())
        }), 400
    
    # In Vercel serverless functions, predictors might be cleared due to cold starts,
    # so we shouldn't immediately return a 503 if the model isn't in memory yet.
    
    old_model = current_model
    
    # Lazy load if not already in memory
    if model_id not in predictors:
        logger.info(f"Loading model {model_id} into memory...")
        predictor = initialize_predictor(AVAILABLE_MODELS[model_id]['type'], use_calibrated=True)
        if predictor:
            predictors[model_id] = predictor
            
            # To aggressively save memory, remove the old model from RAM
            # ONLY doing this because of Render's 512mb strict limits
            if old_model in predictors and old_model != model_id:
                del predictors[old_model]
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to load model',
                'details': f'Could not initialize {AVAILABLE_MODELS[model_id]["name"]}'
            }), 500

    current_model = model_id
    predictor = get_current_predictor()
    
    logger.info(f"Model switched: {old_model} → {current_model}")
    
    return jsonify({
        'success': True,
        'message': 'Model switched successfully',
        'previous_model': old_model,
        'current_model': current_model,
        'model_name': predictor.model_name
    }), 200


@app.route('/models/compare', methods=['POST'])
def compare_models():
    """Compare predictions across all models for the same input."""
    return jsonify({
        'success': False,
        'error': 'Disabled to conserve memory',
        'details': 'Feature disabled on Render Free Tier to avoid Out Of Memory errors resulting from concurrently loading multiple heavy machine learning models.'
    }), 503


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'details': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'details': str(error)
    }), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🏥 CKD PREDICTION API SERVER")
    print("=" * 60)
    print(f"\n✓ Loaded Models: {len(predictors)}/{len(AVAILABLE_MODELS)}")
    for model_key, predictor in predictors.items():
        print(f"   - {AVAILABLE_MODELS[model_key]['name']}: ✓")
    print(f"\n✓ Current Model: {AVAILABLE_MODELS[current_model]['name']}")
    print(f"✓ Status: {'Ready' if predictors else 'Error - No models loaded'}")
    print("\n📡 Starting Flask server...")
    print("   URL: http://localhost:5000")
    print("   Documentation: http://localhost:5000")
    print("\n💡 Press CTRL+C to stop the server\n")
    print("=" * 60 + "\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False  # Set to False in production
    )
