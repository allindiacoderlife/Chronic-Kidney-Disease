import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { AlertCircle, CheckCircle2, Activity } from 'lucide-react'

const PredictionResult = ({ result }) => {
  if (!result) return null

  const isCKD = result.prediction === 'ckd'
  const confidence = result.confidence * 100
  const ckdProbability = result.probabilities.ckd * 100
  const notCkdProbability = result.probabilities.not_ckd * 100

  const getRiskLevel = (probability) => {
    if (probability >= 80) return { level: 'High', color: 'bg-red-500' }
    if (probability >= 50) return { level: 'Moderate', color: 'bg-orange-500' }
    return { level: 'Low', color: 'bg-green-500' }
  }

  const riskLevel = getRiskLevel(ckdProbability)

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      {/* Main Result Alert */}
      <Alert className={isCKD ? 'border-red-500 bg-red-50' : 'border-green-500 bg-green-50'}>
        {isCKD ? (
          <AlertCircle className="h-5 w-5 text-red-600" />
        ) : (
          <CheckCircle2 className="h-5 w-5 text-green-600" />
        )}
        <AlertTitle className="text-lg font-semibold">
          {isCKD ? 'CKD Risk Detected' : 'No CKD Risk Detected'}
        </AlertTitle>
        <AlertDescription className="mt-2">
          {isCKD ? (
            <span>
              The model predicts a <strong>{riskLevel.level.toLowerCase()} risk</strong> of Chronic Kidney Disease based on the provided data.
              We recommend consulting with a healthcare professional for proper diagnosis and treatment.
            </span>
          ) : (
            <span>
              The model indicates a <strong>low probability</strong> of Chronic Kidney Disease based on the provided data.
              Continue regular health monitoring and maintain a healthy lifestyle.
            </span>
          )}
        </AlertDescription>
      </Alert>

      {/* Detailed Results */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Confidence Score */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Confidence Score
            </CardTitle>
            <CardDescription>Model's confidence in this prediction</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-3xl font-bold">{confidence.toFixed(1)}%</span>
                <Badge variant={confidence >= 80 ? 'default' : 'secondary'} className="text-sm">
                  {confidence >= 80 ? 'High Confidence' : 'Moderate Confidence'}
                </Badge>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="bg-blue-600 h-3 rounded-full transition-all duration-700 ease-out"
                  style={{ width: `${confidence}%` }}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Risk Level */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertCircle className="h-5 w-5" />
              Risk Assessment
            </CardTitle>
            <CardDescription>CKD probability classification</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-3xl font-bold">{riskLevel.level}</span>
                <Badge className={`${riskLevel.color} text-white`}>
                  {ckdProbability.toFixed(1)}% CKD Risk
                </Badge>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className={`${riskLevel.color} h-3 rounded-full transition-all duration-700 ease-out`}
                  style={{ width: `${ckdProbability}%` }}
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Probability Breakdown */}
      <Card>
        <CardHeader>
          <CardTitle>Probability Breakdown</CardTitle>
          <CardDescription>Detailed probability analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* CKD Probability */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="font-medium text-sm">Chronic Kidney Disease (CKD)</span>
                <span className="font-bold text-sm">{ckdProbability.toFixed(2)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div
                  className="bg-red-500 h-2.5 rounded-full transition-all duration-700 ease-out"
                  style={{ width: `${ckdProbability}%` }}
                />
              </div>
            </div>

            {/* No CKD Probability */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="font-medium text-sm">No Chronic Kidney Disease</span>
                <span className="font-bold text-sm">{notCkdProbability.toFixed(2)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div
                  className="bg-green-500 h-2.5 rounded-full transition-all duration-700 ease-out"
                  style={{ width: `${notCkdProbability}%` }}
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Model Information */}
      <Card>
        <CardHeader>
          <CardTitle>Model Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Model Used:</span>
            <span className="font-medium">{result.model}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Prediction Class:</span>
            <span className="font-medium">{result.predicted_class}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Timestamp:</span>
            <span className="font-medium">{new Date(result.timestamp).toLocaleString()}</span>
          </div>
        </CardContent>
      </Card>

      {/* Disclaimer */}
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Medical Disclaimer</AlertTitle>
        <AlertDescription className="text-sm">
          This prediction is generated by a machine learning model and should not be used as a definitive diagnosis.
          Always consult with qualified healthcare professionals for medical advice, diagnosis, and treatment.
          Regular health check-ups and professional medical evaluation are essential for accurate diagnosis.
        </AlertDescription>
      </Alert>
    </div>
  )
}

export default PredictionResult
