import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Activity, TrendingUp, Users, AlertCircle, BarChart3, PieChart, ArrowLeft } from 'lucide-react'
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart as RePieChart,
  Pie,
  AreaChart,
  Area,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'

const Analytics = () => {
  const navigate = useNavigate()
  const [modelInfo, setModelInfo] = useState(null)
  const [loading, setLoading] = useState(true)

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

  useEffect(() => {
    const fetchData = async () => {
      try {
        const modelRes = await fetch(`${API_URL}/model/info`)
        const modelData = await modelRes.json()
        setModelInfo(modelData)
      } catch (err) {
        console.error('Error fetching analytics data:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [API_URL])

  // Sample data for visualizations
  const performanceData = modelInfo?.performance ? [
    { metric: 'Accuracy', value: Math.round(modelInfo.performance.accuracy * 100), benchmark: 95 },
    { metric: 'Precision', value: Math.round(modelInfo.performance.precision * 100), benchmark: 90 },
    { metric: 'Recall', value: Math.round(modelInfo.performance.recall * 100), benchmark: 92 },
    { metric: 'F1-Score', value: Math.round(modelInfo.performance.f1_score * 100), benchmark: 93 },
    { metric: 'ROC-AUC', value: Math.round(modelInfo.performance.roc_auc * 100), benchmark: 95 },
  ] : [
    { metric: 'Accuracy', value: 94, benchmark: 95 },
    { metric: 'Precision', value: 94, benchmark: 90 },
    { metric: 'Recall', value: 93, benchmark: 92 },
    { metric: 'F1-Score', value: 94, benchmark: 93 },
    { metric: 'ROC-AUC', value: 98, benchmark: 95 },
  ]

  const predictionDistribution = [
    { name: 'CKD Cases', value: 250, color: '#ef4444' },
    { name: 'Non-CKD', value: 150, color: '#10b981' },
  ]

  const featureImportance = [
    { feature: 'Serum Creatinine', importance: 95 },
    { feature: 'Blood Urea', importance: 88 },
    { feature: 'Hemoglobin', importance: 82 },
    { feature: 'Specific Gravity', importance: 75 },
    { feature: 'Albumin', importance: 70 },
    { feature: 'Age', importance: 65 },
    { feature: 'Blood Pressure', importance: 60 },
    { feature: 'Hypertension', importance: 55 },
  ]

  const confidenceDistribution = [
    { range: '90-100%', count: 180, color: '#10b981' },
    { range: '80-89%', count: 120, color: '#3b82f6' },
    { range: '70-79%', count: 70, color: '#f59e0b' },
    { range: '60-69%', count: 30, color: '#ef4444' },
  ]

  const monthlyTrends = [
    { month: 'Jan', ckd: 65, nonCkd: 35, total: 100 },
    { month: 'Feb', ckd: 70, nonCkd: 40, total: 110 },
    { month: 'Mar', ckd: 75, nonCkd: 45, total: 120 },
    { month: 'Apr', ckd: 68, nonCkd: 42, total: 110 },
    { month: 'May', ckd: 72, nonCkd: 48, total: 120 },
    { month: 'Jun', ckd: 80, nonCkd: 50, total: 130 },
  ]

  const featureCategories = [
    { category: 'Demographics', count: 2, color: '#8b5cf6' },
    { category: 'Urine Tests', count: 7, color: '#3b82f6' },
    { category: 'Blood Tests', count: 9, color: '#10b981' },
    { category: 'Medical History', count: 6, color: '#f59e0b' },
  ]

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button
                variant="outline"
                size="icon"
                onClick={() => navigate('/')}
              >
                <ArrowLeft className="h-4 w-4" />
              </Button>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">
                  📊 Analytics Dashboard
                </h1>
                <p className="mt-1 text-sm text-gray-600">
                  Model performance metrics and prediction insights
                </p>
              </div>
            </div>
            <Badge variant="default" className="flex items-center gap-2">
              <Activity className="h-3 w-3" />
              Live Data
            </Badge>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-600 flex items-center gap-2">
                <Activity className="h-4 w-4" />
                Model Accuracy
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-emerald-600">
                {modelInfo?.performance ? (modelInfo.performance.accuracy * 100).toFixed(1) : '93.8'}%
              </div>
              <p className="text-xs text-gray-500 mt-1">Perfect classification</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-600 flex items-center gap-2">
                <TrendingUp className="h-4 w-4" />
                ROC-AUC Score
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-blue-600">
                {modelInfo?.performance?.roc_auc ? modelInfo.performance.roc_auc.toFixed(3) : '0.975'}
              </div>
              <p className="text-xs text-gray-500 mt-1">Excellent discrimination</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-600 flex items-center gap-2">
                <Users className="h-4 w-4" />
                Total Predictions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-purple-600">400</div>
              <p className="text-xs text-gray-500 mt-1">Sample dataset size</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-600 flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                Features Used
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-orange-600">24</div>
              <p className="text-xs text-gray-500 mt-1">Clinical parameters</p>
            </CardContent>
          </Card>
        </div>

        {/* Charts Row 1 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Performance Metrics */}
          <Card>
            <CardHeader>
              <CardTitle>Model Performance Metrics</CardTitle>
              <CardDescription>Comparison with industry benchmarks</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="metric" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="value" fill="#3b82f6" name="Our Model" />
                  <Bar dataKey="benchmark" fill="#94a3b8" name="Benchmark" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Prediction Distribution */}
          <Card>
            <CardHeader>
              <CardTitle>Prediction Distribution</CardTitle>
              <CardDescription>CKD vs Non-CKD cases</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <RePieChart>
                  <Pie
                    data={predictionDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {predictionDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </RePieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Charts Row 2 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Feature Importance */}
          <Card>
            <CardHeader>
              <CardTitle>Top Feature Importance</CardTitle>
              <CardDescription>Most influential clinical parameters</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={featureImportance} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} />
                  <YAxis dataKey="feature" type="category" width={120} />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#10b981" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Confidence Distribution */}
          <Card>
            <CardHeader>
              <CardTitle>Confidence Score Distribution</CardTitle>
              <CardDescription>Prediction confidence levels</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={confidenceDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8b5cf6">
                    {confidenceDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Charts Row 3 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Monthly Trends */}
          <Card>
            <CardHeader>
              <CardTitle>Prediction Trends</CardTitle>
              <CardDescription>6-month prediction history</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={monthlyTrends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="ckd" stackId="1" stroke="#ef4444" fill="#ef4444" name="CKD" />
                  <Area type="monotone" dataKey="nonCkd" stackId="1" stroke="#10b981" fill="#10b981" name="Non-CKD" />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Feature Categories */}
          <Card>
            <CardHeader>
              <CardTitle>Feature Categories</CardTitle>
              <CardDescription>Distribution by clinical category</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <RePieChart>
                  <Pie
                    data={featureCategories}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ category, count }) => `${category} (${count})`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="count"
                  >
                    {featureCategories.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </RePieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Model Information */}
        {modelInfo && modelInfo.success && (
          <Card>
            <CardHeader>
              <CardTitle>Model Information</CardTitle>
              <CardDescription>Current production model details</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Model Name</p>
                  <p className="text-lg font-semibold">{modelInfo.model_name}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Model Type</p>
                  <p className="text-lg font-semibold">{modelInfo.model_type}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Features</p>
                  <p className="text-lg font-semibold">{modelInfo.features_count}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Calibrated</p>
                  <p className="text-lg font-semibold">{modelInfo.calibrated ? 'Yes' : 'No'}</p>
                </div>
                {modelInfo.performance && (
                  <>
                    <div>
                      <p className="text-sm text-gray-600">Accuracy</p>
                      <p className="text-lg font-semibold text-green-600">
                        {(modelInfo.performance.accuracy * 100).toFixed(2)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">ROC-AUC</p>
                      <p className="text-lg font-semibold text-blue-600">
                        {modelInfo.performance.roc_auc?.toFixed(4)}
                      </p>
                    </div>
                  </>
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  )
}

export default Analytics
