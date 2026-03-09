import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import PredictionForm from "@/components/PredictionForm";
import PredictionResult from "@/components/PredictionResult";
import ModelSelector from "@/components/ModelSelector";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Activity, AlertCircle, CheckCircle2, Server, BarChart3, Brain } from "lucide-react";

const Home = () => {
  const navigate = useNavigate();
  const [health, setHealth] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("predict");
  const [currentModelName, setCurrentModelName] = useState(null);

  // Use Vite environment variable if provided, otherwise default to localhost:5000
  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

  useEffect(() => {
    let mounted = true;

    const checkHealth = async () => {
      try {
        console.log("Checking API health at:", `${API_URL}/health`);
        const res = await fetch(`${API_URL}/health`, {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
          },
        });
        console.log("API response status:", res.status, res.ok);
        const json = await res.json();
        console.log("API response data:", json);
        if (mounted) {
          setHealth({ ok: res.ok, ...json });
          if (json.model_name) {
            setCurrentModelName(json.model_name);
          }
          console.log("API connected successfully");
        }
      } catch (err) {
        console.error("API connection error:", err);
        if (mounted) {
          setHealth({ ok: false, error: err.message });
          console.log("API connection failed:", err.message);
        }
      }
    };

    checkHealth();

    return () => {
      mounted = false;
    };
  }, [API_URL]);

  const handlePredictionSubmit = async (formData) => {
    setIsLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || "Prediction failed");
      }

      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelChange = (modelId, modelName) => {
    setCurrentModelName(modelName);
    setPrediction(null); // Clear previous predictions
    setError(null);

    // Refresh health status
    fetch(`${API_URL}/health`)
      .then(res => res.json())
      .then(json => setHealth({ ok: true, ...json }))
      .catch(err => console.error("Failed to refresh health:", err));
  };

  return (
    <div className="min-h-screen bg-slate-50 relative selection:bg-indigo-100 selection:text-indigo-900 font-manrope">
      {/* Subtle Background Glows */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none z-0 block">
        <div className="absolute -top-1/2 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-indigo-500/10 rounded-full blur-[120px]" />
        <div className="absolute top-1/4 -right-1/4 w-[600px] h-[600px] bg-emerald-500/10 rounded-full blur-[120px]" />
        <div className="absolute -bottom-1/2 left-0 w-[1000px] h-[1000px] bg-purple-500/10 rounded-full blur-[150px]" />
      </div>

      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-slate-200/60 bg-white/60 backdrop-blur-xl supports-backdrop-blur:bg-white/40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-indigo-50 to-emerald-50 rounded-lg border border-slate-200 shadow-sm">
                <Brain className="h-6 w-6 text-indigo-600" />
              </div>
              <div>
                <h1 className="text-2xl font-medium tracking-tight font-instrument-serif text-slate-900">
                  RenalInsight AI
                </h1>
                <p className="mt-0.5 text-xs text-slate-500 tracking-wider uppercase font-semibold">
                  CKD Risk Assessment Platform
                </p>
              </div>
            </div>
            {health && (
              <div className="flex items-center gap-4">
                <Button
                  variant="outline"
                  onClick={() => navigate('/analytics')}
                  className="flex items-center gap-2 bg-white hover:bg-slate-50 border-slate-200 transition-all font-manrope text-sm h-9 rounded-full px-4 text-slate-700 shadow-sm"
                >
                  <BarChart3 className="h-4 w-4" />
                  Analytics
                </Button>
                <div className="flex items-center justify-center">
                  <Badge
                    variant={health.ok ? "default" : "destructive"}
                    className={`flex items-center gap-2 py-1.5 px-3 rounded-full text-xs font-semibold shadow-sm ${health.ok
                      ? "bg-emerald-50 text-emerald-700 border-emerald-200 border hover:bg-emerald-100"
                      : "bg-red-50 text-red-700 border-red-200 border hover:bg-red-100"
                      }`}
                  >
                    <span className={`relative flex h-2 w-2 mr-0.5`}>
                      {health.ok && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>}
                      <span className={`relative inline-flex rounded-full h-2 w-2 ${health.ok ? "bg-emerald-500" : "bg-red-500"}`}></span>
                    </span>
                    {health.ok ? "API Connected" : "API Offline"}
                  </Badge>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 relative z-10 w-full min-h-[calc(100vh-80px)]">
        {/* API Status Alert */}
        {health && !health.ok && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>API Connection Error</AlertTitle>
            <AlertDescription>
              Cannot connect to the backend API. Please make sure the Flask
              server is running on port 5000.
              <br />
              <span className="text-sm mt-1 block">Error: {health.error}</span>
            </AlertDescription>
          </Alert>
        )}

        {health && health.ok && (
          <Alert className="mb-8 border-emerald-200 bg-emerald-50/80 backdrop-blur-md shadow-lg shadow-emerald-500/5">
            <CheckCircle2 className="h-4 w-4 text-emerald-600" />
            <AlertTitle className="text-emerald-900 font-semibold tracking-wide">System Ready</AlertTitle>
            <AlertDescription className="text-emerald-800 mt-1 font-manrope">
              Model: <strong className="text-emerald-900 font-medium">{currentModelName || health.model_name}</strong> | Status:{" "}
              <strong className="text-emerald-900 font-medium">{health.status}</strong>
              {health.total_models && (
                <span className="ml-3 text-emerald-700">
                  | Loaded: <strong className="text-emerald-900 font-medium">{health.total_models}/{health.available_models}</strong> models
                </span>
              )}
            </AlertDescription>
          </Alert>
        )}

        {/* Information Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <Card className="bg-white/60 backdrop-blur-md border border-slate-200 shadow-xl shadow-indigo-500/5 group hover:bg-white transition-all duration-300">
            <CardHeader className="pb-3">
              <CardTitle className="text-base font-semibold flex items-center gap-3 text-slate-800">
                <div className="p-2 rounded-md bg-indigo-50 text-indigo-600 group-hover:scale-110 group-hover:bg-indigo-100 transition-all border border-indigo-100">
                  <Activity className="h-4 w-4" />
                </div>
                High Accuracy
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-slate-600 leading-relaxed font-manrope">
                Our model achieves exceptional precision using state-of-the-art diagnostic algorithms and rigorous calibration.
              </p>
            </CardContent>
          </Card>

          <Card className="bg-white/60 backdrop-blur-md border border-slate-200 shadow-xl shadow-emerald-500/5 group hover:bg-white transition-all duration-300">
            <CardHeader className="pb-3">
              <CardTitle className="text-base font-semibold flex items-center gap-3 text-slate-800">
                <div className="p-2 rounded-md bg-emerald-50 text-emerald-600 group-hover:scale-110 group-hover:bg-emerald-100 transition-all border border-emerald-100">
                  <CheckCircle2 className="h-4 w-4" />
                </div>
                24 Clinical Features
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-slate-600 leading-relaxed font-manrope">
                Comprehensive patient analysis utilizing robust clinical markers and key laboratory test results.
              </p>
            </CardContent>
          </Card>

          <Card className="bg-white/60 backdrop-blur-md border border-slate-200 shadow-xl shadow-purple-500/5 group hover:bg-white transition-all duration-300">
            <CardHeader className="pb-3">
              <CardTitle className="text-base font-semibold flex items-center gap-3 text-slate-800">
                <div className="p-2 rounded-md bg-purple-50 text-purple-600 group-hover:scale-110 group-hover:bg-purple-100 transition-all border border-purple-100">
                  <Server className="h-4 w-4" />
                </div>
                Real-time Feedback
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-slate-600 leading-relaxed font-manrope">
                Instant predictive endpoints combined with actionable confidence scores for immediate clinical review.
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Main Content - Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-8">
          <TabsList className="grid w-full max-w-md mx-auto grid-cols-2 bg-white/60 backdrop-blur-md p-1 border border-slate-200 rounded-xl shadow-sm">
            <TabsTrigger value="predict" className="flex items-center gap-2 rounded-lg data-[state=active]:bg-white data-[state=active]:text-indigo-900 data-[state=active]:shadow-sm transition-all font-medium py-2.5 text-slate-600">
              <Activity className="h-4 w-4" />
              Diagnostics Form
            </TabsTrigger>
            <TabsTrigger value="models" className="flex items-center gap-2 rounded-lg data-[state=active]:bg-white data-[state=active]:text-indigo-900 data-[state=active]:shadow-sm transition-all font-medium py-2.5 text-slate-600">
              <Brain className="h-4 w-4" />
              Model Configuration
            </TabsTrigger>
          </TabsList>

          {/* Prediction Tab */}
          <TabsContent value="predict" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Left Column - Form */}
              <div>
                <Card className="shadow-2xl shadow-indigo-900/5 bg-white/80 backdrop-blur-xl border-slate-200 relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/5 rounded-full blur-[80px] pointer-events-none" />
                  <CardHeader className="border-b border-slate-100 pb-6">
                    <CardTitle className="text-2xl font-instrument-serif font-normal tracking-wide text-slate-900">Patient Profile</CardTitle>
                    <CardDescription className="text-slate-500 font-manrope">
                      Input the patient clinical and laboratory vitals below.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <PredictionForm
                      onSubmit={handlePredictionSubmit}
                      isLoading={isLoading}
                    />
                  </CardContent>
                </Card>
              </div>

              {/* Right Column - Results */}
              <div>
                <Card className="shadow-2xl shadow-indigo-900/5 bg-white/80 backdrop-blur-xl border-slate-200 sticky top-24 relative overflow-hidden">
                  <div className="absolute -bottom-20 -left-20 w-64 h-64 bg-emerald-500/5 rounded-full blur-[80px] pointer-events-none" />
                  <CardHeader className="border-b border-slate-100 pb-6">
                    <CardTitle className="text-2xl font-instrument-serif font-normal tracking-wide text-slate-900">Diagnostic Output</CardTitle>
                    <CardDescription className="text-slate-500 font-manrope">
                      {prediction
                        ? `Analysis powered by ${currentModelName || 'ML Model'}`
                        : "Awaiting patient vitals to generate predictive assessment."}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {error && (
                      <Alert variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Error</AlertTitle>
                        <AlertDescription>{error}</AlertDescription>
                      </Alert>
                    )}

                    {!prediction && !error && !isLoading && (
                      <div className="text-center py-12 text-gray-500">
                        <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                        <p>
                          Enter patient data and click "Predict CKD Risk" to see
                          results
                        </p>
                      </div>
                    )}

                    {isLoading && (
                      <div className="text-center py-12">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                        <p className="text-gray-600">Analyzing patient data...</p>
                      </div>
                    )}

                    {prediction && <PredictionResult result={prediction} />}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Model Selection Tab */}
          <TabsContent value="models">
            <ModelSelector apiUrl={API_URL} onModelChange={handleModelChange} />
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="mt-20 border-t border-slate-200 bg-white/40 backdrop-blur-sm relative z-10 w-full relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex items-center justify-between opacity-80">
          <p className="text-xs text-slate-500 font-manrope">
            © 2025 RenalInsight Platform. Developed for clinical research.
          </p>
          <div className="flex gap-4">
            <span className="text-xs text-slate-500 hover:text-slate-700 transition-colors font-manrope cursor-pointer">Privacy</span>
            <span className="text-xs text-slate-500 hover:text-slate-700 transition-colors font-manrope cursor-pointer">Terms</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Home;
