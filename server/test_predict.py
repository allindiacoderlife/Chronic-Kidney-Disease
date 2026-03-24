import json
from api.predict import CKDPredictor

predictor = CKDPredictor()
predictor.load_model('random_forest', use_calibrated=True)

data1 = {
    'age': "48", 'bp': "80", 'sg': "1.020", 'al': "1", 'su': "0",
    'rbc': 'abnormal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
    'bgr': "121", 'bu': "36", 'sc': "1.2", 'sod': "138", 'pot': "4.4",
    'hemo': "15.4", 'pcv': "44", 'wc': "7800", 'rc': "5.2",
    'htn': 'yes', 'dm': 'yes', 'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no'
}

data2 = data1.copy()
data2['age'] = "10"
data2['bp'] = "150"
data2['htn'] = "no"

X1 = predictor.preprocess_input(data1)
X2 = predictor.preprocess_input(data2)

p1 = predictor.predict_single(data1)
p2 = predictor.predict_single(data2)

print('X1 == X2:', (X1 == X2).all())
diffs = [f for i, f in enumerate(predictor.feature_names) if X1[0, i] != X2[0, i]]
print('Diff features:', diffs)

print('\nP1 proba:', p1['probabilities'])
print('P2 proba:', p2['probabilities'])
