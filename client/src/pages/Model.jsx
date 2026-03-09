import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import PredictionForm from "@/components/PredictionForm";
import PredictionResult from "@/components/PredictionResult";
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
import { Activity, AlertCircle, CheckCircle2, Server, BarChart3 } from "lucide-react";

const Home = () => {
  const navigate = useNavigate();
  const [health, setHealth] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

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

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                🏥 Chronic Kidney Disease Prediction
              </h1>
              <p className="mt-1 text-sm text-gray-600">
                AI-powered CKD risk assessment using machine learning
              </p>
            </div>
            {health && (
              <div className="flex items-center gap-3">
                <Button
                  variant="outline"
                  onClick={() => navigate('/analytics')}
                  className="flex items-center gap-2"
                >
                  <BarChart3 className="h-4 w-4" />
                  Analytics
                </Button>
                <Badge
                  variant={health.ok ? "default" : "destructive"}
                  className="flex items-center gap-2"
                >
                  <Server className="h-3 w-3" />
                  {health.ok ? "API Connected" : "API Offline"}
                </Badge>
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
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
          <Alert className="mb-6 border-green-500 bg-green-50">
            <CheckCircle2 className="h-4 w-4 text-green-600" />
            <AlertTitle className="text-green-900">System Ready</AlertTitle>
            <AlertDescription className="text-green-800">
              Model: <strong>{health.model_name}</strong> | Status:{" "}
              <strong>{health.status}</strong>
            </AlertDescription>
          </Alert>
        )}

        {/* Information Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <Activity className="h-5 w-5 text-blue-600" />
                High Accuracy
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">
                Our model achieves 100% accuracy with advanced calibration
                techniques
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-600" />
                24 Features
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">
                Comprehensive analysis using clinical and laboratory test
                results
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <Server className="h-5 w-5 text-purple-600" />
                Real-time
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">
                Instant predictions with confidence scores and risk assessment
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Form */}
          <div>
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="text-2xl">Patient Data Entry</CardTitle>
                <CardDescription>
                  Fill in the patient's medical information to get a CKD risk
                  prediction
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
            <Card className="shadow-lg sticky top-8">
              <CardHeader>
                <CardTitle className="text-2xl">Prediction Results</CardTitle>
                <CardDescription>
                  {prediction
                    ? "Analysis complete"
                    : "Results will appear here after prediction"}
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
      </main>

      {/* Footer */}
      <footer className="mt-16 bg-gray-50 border-t">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-600">
            CKD Prediction System © 2025 | For educational and research purposes
            only
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Home;
