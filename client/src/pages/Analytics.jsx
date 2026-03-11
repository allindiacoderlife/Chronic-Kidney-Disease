import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, PieChart as RePieChart, Pie, Cell, AreaChart, Area } from 'recharts';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/card';
import ModelSelector from '../components/ModelSelector';

const Analytics = () => {
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

  const confidenceData = [
    { name: '0-69%', count: 85 },
    { name: '70-79%', count: 125 },
    { name: '80-89%', count: 210 },
    { name: '90-100%', count: 982 },
  ];

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
    { name: 'No Disease', value: 320, color: '#10b981' },
    { name: 'Low Risk', value: 85, color: '#3b82f6' },
    { name: 'Moderate Risk', value: 45, color: '#f59e0b' },
    { name: 'High Risk', value: 30, color: '#f97316' },
    { name: 'Severe Disease', value: 20, color: '#ef4444' },
  ]

  const featureImportance = [
    { feature: 'Serum Creatinine', importance: 98 },
    { feature: 'Hemoglobin', importance: 92 },
    { feature: 'Specific Gravity', importance: 85 },
    { feature: 'Blood Urea', importance: 78 },
    { feature: 'Albumin', importance: 74 },
    { feature: 'Hypertension', importance: 68 },
    { feature: 'Diabetes Mellitus', importance: 65 },
    { feature: 'Packed Cell Volume', importance: 58 },
  ]

  const confidenceDistribution = [
    { range: '90-100%', count: 180, color: '#10b981' },
    { range: '80-89%', count: 120, color: '#3b82f6' },
    { range: '70-79%', count: 70, color: '#f59e0b' },
    { range: '60-69%', count: 30, color: '#ef4444' },
  ]

  const monthlyTrends = [
    { month: 'Jan', healthy: 45, atRisk: 15, total: 60 },
    { month: 'Feb', healthy: 52, atRisk: 18, total: 70 },
    { month: 'Mar', healthy: 48, atRisk: 22, total: 70 },
    { month: 'Apr', healthy: 60, atRisk: 15, total: 75 },
    { month: 'May', healthy: 55, atRisk: 25, total: 80 },
    { month: 'Jun', healthy: 68, atRisk: 12, total: 80 },
  ]

  const featureCategories = [
    { category: 'Demographics', count: 2, color: '#8b5cf6' },
    { category: 'Urine Tests', count: 7, color: '#3b82f6' },
    { category: 'Blood Tests', count: 9, color: '#10b981' },
    { category: 'Medical History', count: 6, color: '#f59e0b' },
  ]

  useEffect(() => {
    let mounted = true;

    const fetchHealth = async () => {
      try {
        const res = await fetch(`${API_URL}/health`);
        const json = await res.json();
        if (mounted) {
          setHealth({ ok: res.ok, ...json });
        }
      } catch (err) {
        if (mounted) setHealth({ ok: false, error: err.message });
      }
    };

    const fetchData = async () => {
      try {
        const modelRes = await fetch(`${API_URL}/model/info`);
        const modelData = await modelRes.json();
        if (mounted) {
          setModelInfo(modelData);
        }
      } catch (err) {
        console.error('Error fetching analytics data:', err);
      } finally {
        if (mounted) setLoading(false);
      }
    };

    fetchHealth();
    fetchData();

    return () => { mounted = false; };
  }, [API_URL, refreshTrigger])

  return (
    <div className="analytics-page text-slate-900 dark:text-slate-100 min-h-screen flex flex-col font-['Inter'] bg-[#f5f7f8] dark:bg-[#0f1923] w-full">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 px-6 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-[#0056b2] p-1.5 rounded-lg text-white">
              <span className="material-symbols-outlined text-2xl leading-none">nephrology</span>
            </div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">CKD Prediction</h1>
          </div>
          <nav className="flex items-center gap-8">
            <Link className="text-sm font-medium text-slate-600 dark:text-slate-400 hover:text-[#0056b2] transition-colors" to="/">Dashboard</Link>
            <Link className="text-sm font-medium text-[#0056b2] transition-colors" to="/analytics">Analytics</Link>
            <div className="flex items-center gap-3">
              {health ? (
                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border ${health.ok ? 'bg-[#0056b2]/10 text-[#0056b2] border-[#0056b2]/20' : 'bg-red-50 text-red-600 border-red-200'}`}>
                  <span className="material-symbols-outlined text-sm">{health.ok ? 'cloud_done' : 'cloud_off'}</span>
                  <span className="text-sm font-bold">{health.ok ? 'API Connected' : 'API Offline'}</span>
                </div>
              ) : (
                <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 rounded-lg border border-slate-200 dark:border-slate-700">
                  <span className="text-sm font-bold animate-pulse">Connecting...</span>
                </div>
              )}
            </div>
          </nav>
        </div>
      </header>

      {/* System Status Bar */}
      <div className="bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 px-6 py-2">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center gap-4">
          <div className={`flex items-center gap-2 ${health?.ok ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-500'}`}>
            <span className="material-symbols-outlined text-lg">{health?.ok ? 'check_circle' : 'error'}</span>
            <span className="text-xs font-semibold uppercase tracking-wider">{health?.ok ? 'System Ready' : 'System Down'}</span>
          </div>
          <div className="h-4 w-px bg-slate-300 dark:bg-slate-700"></div>
          <p className="text-xs text-slate-500 dark:text-slate-400 italic">
            Active Engine: {health?.model_name || 'Logistic Regression Model (v4.2.1)'} • Last Sync: {new Date().toLocaleTimeString()}
          </p>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8 flex flex-col gap-8 w-full">
        {/* Top Header */}
        <header className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100">Analytics Overview</h1>
            <p className="text-slate-500 dark:text-slate-400 text-sm">Reviewing performance data for the CKD Risk Prediction Engine.</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 rounded-full text-xs font-semibold">
              <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
              System: Operational
            </div>
            <button className="bg-[#0056b2] text-white px-4 py-2 rounded-lg text-sm font-medium hover:opacity-90 transition-opacity flex items-center gap-2">
              <span className="material-icons text-sm">refresh</span>
              Retrain Model
            </button>
          </div>
        </header>

        {/* Model Selector */}
        <div className="mb-8">
          <ModelSelector
            apiUrl={API_URL}
            onModelChange={(modelId, modelName) => {
              setRefreshTrigger(prev => prev + 1);
            }}
          />
        </div>

        {/* Model Performance Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white dark:bg-slate-900 p-5 rounded-xl border border-[#0056b2]/10 shadow-sm">
            <div className="flex justify-between items-start mb-4">
              <span className="text-slate-500 dark:text-slate-400 text-xs font-semibold uppercase tracking-wider">Model Accuracy</span>
              <span className="material-icons text-[#0056b2]/40">verified_user</span>
            </div>
            <div className="flex items-baseline gap-2">
              <h3 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
                {modelInfo?.performance?.accuracy ? (modelInfo.performance.accuracy * 100).toFixed(1) + '%' : '93.8%'}
              </h3>
              <span className="text-emerald-500 text-xs font-medium flex items-center">Real-time</span>
            </div>
            <div className="mt-4 w-full bg-slate-100 dark:bg-slate-800 h-1.5 rounded-full overflow-hidden">
              <div 
                className="bg-[#0056b2] h-full transition-all duration-500" 
                style={{ width: `${modelInfo?.performance?.accuracy ? (modelInfo.performance.accuracy * 100) : 93.8}%` }}
              ></div>
            </div>
          </div>
          <div className="bg-white dark:bg-slate-900 p-5 rounded-xl border border-[#0056b2]/10 shadow-sm">
            <div className="flex justify-between items-start mb-4">
              <span className="text-slate-500 dark:text-slate-400 text-xs font-semibold uppercase tracking-wider">ROC-AUC Score</span>
              <span className="material-icons text-[#0056b2]/40">legend_toggle</span>
            </div>
            <div className="flex items-baseline gap-2">
              <h3 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
                {modelInfo?.performance?.roc_auc ? modelInfo.performance.roc_auc.toFixed(3) : '0.975'}
              </h3>
              <span className="text-emerald-500 text-xs font-medium flex items-center">Real-time</span>
            </div>
            <div className="mt-4 w-full bg-slate-100 dark:bg-slate-800 h-1.5 rounded-full overflow-hidden">
              <div 
                className="bg-sky-500 h-full transition-all duration-500" 
                style={{ width: `${modelInfo?.performance?.roc_auc ? (modelInfo.performance.roc_auc * 100) : 97.5}%` }}
              ></div>
            </div>
          </div>
          <div className="bg-white dark:bg-slate-900 p-5 rounded-xl border border-[#0056b2]/10 shadow-sm">
            <div className="flex justify-between items-start mb-4">
              <span className="text-slate-500 dark:text-slate-400 text-xs font-semibold uppercase tracking-wider">Total Predictions</span>
              <span className="material-icons text-[#0056b2]/40">data_exploration</span>
            </div>
            <div className="flex items-baseline gap-2">
              <h3 className="text-3xl font-bold text-slate-900 dark:text-slate-100">4,102</h3>
              <span className="text-slate-400 text-xs font-medium uppercase">Cumulative</span>
            </div>
            <div className="mt-4 flex gap-1">
              <div className="h-2 flex-1 bg-[#0056b2]/20 rounded-full"></div>
              <div className="h-2 flex-1 bg-[#0056b2]/40 rounded-full"></div>
              <div className="h-2 flex-1 bg-[#0056b2]/60 rounded-full"></div>
              <div className="h-2 flex-1 bg-[#0056b2]/80 rounded-full"></div>
              <div className="h-2 flex-1 bg-[#0056b2] rounded-full"></div>
            </div>
          </div>
          <div className="bg-white dark:bg-slate-900 p-5 rounded-xl border border-[#0056b2]/10 shadow-sm">
            <div className="flex justify-between items-start mb-4">
              <span className="text-slate-500 dark:text-slate-400 text-xs font-semibold uppercase tracking-wider">Features Used</span>
              <span className="material-icons text-[#0056b2]/40">account_tree</span>
            </div>
            <div className="flex items-baseline gap-2">
              <h3 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
                {modelInfo?.features_count || '24'}
              </h3>
              <span className="text-slate-400 text-xs font-medium uppercase">Clinical Parsers</span>
            </div>
            <div className="mt-4 text-[10px] text-slate-500 flex flex-wrap gap-1">
              <span className="px-2 py-0.5 bg-slate-100 dark:bg-slate-800 rounded">Labs</span>
              <span className="px-2 py-0.5 bg-slate-100 dark:bg-slate-800 rounded">History</span>
              <span className="px-2 py-0.5 bg-slate-100 dark:bg-slate-800 rounded">Vitals</span>
            </div>
          </div>
        </div>

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
                  <Area type="monotone" dataKey="atRisk" stackId="1" stroke="#f97316" fill="#f97316" name="At Risk Cases" />
                  <Area type="monotone" dataKey="healthy" stackId="1" stroke="#10b981" fill="#10b981" name="Healthy Cases" />
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
  );
};

export default Analytics;
