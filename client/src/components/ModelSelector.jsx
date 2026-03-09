import React, { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Brain,
  RefreshCw,
  CheckCircle2,
  TrendingUp,
  Zap,
  Network,
  Layers,
  Activity,
} from "lucide-react";

const MODEL_ICONS = {
  logistic_regression: Brain,
  random_forest: Layers,
  xgboost: TrendingUp,
  lightgbm: Zap,
  mlp: Network,
};

const ModelSelector = ({ apiUrl, onModelChange }) => {
  const [models, setModels] = useState([]);
  const [currentModel, setCurrentModel] = useState(null);
  const [loading, setLoading] = useState(false);
  const [switching, setSwitching] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${apiUrl}/models`);
      const data = await response.json();

      if (data.success) {
        setModels(data.models);
        setCurrentModel(data.current_model);
      } else {
        setError(data.error || "Failed to fetch models");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const switchModel = async (modelId) => {
    setSwitching(modelId);
    setError(null);

    try {
      const response = await fetch(`${apiUrl}/models/${modelId}`, {
        method: "POST",
      });

      const data = await response.json();

      if (data.success) {
        setCurrentModel(modelId);
        
        // Notify parent component
        if (onModelChange) {
          onModelChange(modelId, data.model_name);
        }
      } else {
        setError(data.error || "Failed to switch model");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setSwitching(null);
    }
  };

  const formatPerformance = (perf) => {
    if (!perf) return null;
    return {
      accuracy: (perf.accuracy * 100).toFixed(1),
      precision: (perf.precision * 100).toFixed(1),
      recall: (perf.recall * 100).toFixed(1),
      f1: (perf.f1_score * 100).toFixed(1),
      rocAuc: (perf.roc_auc * 100).toFixed(1),
    };
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 animate-pulse" />
            Loading Models...
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-gray-500">Fetching available models...</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-blue-600" />
                Model Selection
              </CardTitle>
              <CardDescription>
                Choose from {models.length} trained ML models
              </CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={fetchModels}
              disabled={loading}
            >
              <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {models.map((model) => {
              const Icon = MODEL_ICONS[model.id] || Brain;
              const perf = formatPerformance(model.performance);
              const isActive = model.id === currentModel;
              const isSwitching = switching === model.id;

              return (
                <Card
                  key={model.id}
                  className={`cursor-pointer transition-all ${
                    isActive
                      ? "ring-2 ring-blue-500 bg-blue-50"
                      : "hover:shadow-lg"
                  } ${!model.loaded ? "opacity-50" : ""}`}
                  onClick={() => model.loaded && !isSwitching && switchModel(model.id)}
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-2">
                        <Icon className={`h-5 w-5 ${isActive ? "text-blue-600" : "text-gray-600"}`} />
                        <div>
                          <CardTitle className="text-base">
                            {model.name}
                          </CardTitle>
                        </div>
                      </div>
                      {isActive && (
                        <Badge variant="default" className="flex items-center gap-1">
                          <CheckCircle2 className="h-3 w-3" />
                          Active
                        </Badge>
                      )}
                      {!model.loaded && (
                        <Badge variant="destructive">Not Loaded</Badge>
                      )}
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-xs text-gray-600 mb-3">
                      {model.description}
                    </p>

                    {perf && (
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-600">Accuracy:</span>
                          <span className="font-semibold">{perf.accuracy}%</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-600">Precision:</span>
                          <span className="font-semibold">{perf.precision}%</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-600">Recall:</span>
                          <span className="font-semibold">{perf.recall}%</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-600">ROC-AUC:</span>
                          <span className="font-semibold">{perf.rocAuc}%</span>
                        </div>
                      </div>
                    )}

                    {model.loaded && !isActive && (
                      <Button
                        className="w-full mt-3"
                        variant="outline"
                        size="sm"
                        disabled={isSwitching}
                      >
                        {isSwitching ? (
                          <>
                            <RefreshCw className="h-3 w-3 mr-2 animate-spin" />
                            Switching...
                          </>
                        ) : (
                          "Switch to this model"
                        )}
                      </Button>
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ModelSelector;
