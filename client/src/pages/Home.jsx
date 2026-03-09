import React, { useState, useEffect } from "react";

const Home = () => {
  const [health, setHealth] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

  // Form State initialized with defaults to cover all 24 CKD features
  const [formData, setFormData] = useState({
    age: "",
    bp: "",
    sg: "1.020",
    al: "0",
    su: "0",
    rbc: "normal",
    pc: "normal",
    pcc: "notpresent",
    ba: "notpresent",
    bgr: "",
    bu: "",
    sc: "",
    sod: "",
    pot: "",
    hemo: "",
    pcv: "",
    wc: "",
    rc: "",
    htn: "no",
    dm: "no",
    cad: "no",
    appet: "good",
    pe: "no",
    ane: "no",
  });

  useEffect(() => {
    let mounted = true;
    const checkHealth = async () => {
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
    checkHealth();
    return () => { mounted = false; };
  }, [API_URL]);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? (checked ? "yes" : "no") : value,
    }));
  };

  const handlePredict = async () => {
    setIsLoading(true);
    setError(null);
    setPrediction(null);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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

  const fillSampleData = () => {
    setFormData({
      age: "62", bp: "90", sg: "1.010", al: "3", su: "2",
      rbc: "abnormal", pc: "abnormal", pcc: "present", ba: "notpresent",
      bgr: "423", bu: "53", sc: "1.8", sod: "114", pot: "4",
      hemo: "9.6", pcv: "31", wc: "7500", rc: "4",
      htn: "yes", dm: "yes", cad: "yes", appet: "poor", pe: "yes", ane: "yes"
    });
  };

  return (
    <div className="bg-[#f5f7f8] dark:bg-[#0f1923] font-display text-slate-900 dark:text-slate-100 min-h-screen">

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
            <a className="text-sm font-medium text-slate-600 dark:text-slate-400 hover:text-[#0056b2] transition-colors" href="#">Dashboard</a>
            <a className="text-sm font-medium text-slate-600 dark:text-slate-400 hover:text-[#0056b2] transition-colors" href="/analytics">Analytics</a>
            <div className="flex items-center gap-3">
              {health ? (
                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border ${health.ok ? 'bg-[#0056b2]/10 text-[#0056b2] border-[#0056b2]/20' : 'bg-red-50 text-red-600 border-red-200'}`}>
                  <span className="material-symbols-outlined text-sm">{health.ok ? 'cloud_done' : 'cloud_off'}</span>
                  <span className="text-sm font-bold">{health.ok ? 'API Connected' : 'API Offline'}</span>
                </div>
              ) : (
                <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 text-slate-500 rounded-lg border border-slate-200">
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

      <main className="max-w-7xl mx-auto px-6 py-8 flex flex-col gap-8">

        {/* Key Metrics Row */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-slate-800 p-5 rounded-xl border border-slate-200 dark:border-slate-700 flex items-start gap-4">
            <div className="p-3 bg-[#0056b2]/10 text-[#0056b2] rounded-lg">
              <span className="material-symbols-outlined">target</span>
            </div>
            <div>
              <h3 className="font-bold text-slate-900 dark:text-white">High Accuracy</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400">Model validated on 15k+ clinical records</p>
            </div>
          </div>
          <div className="bg-white dark:bg-slate-800 p-5 rounded-xl border border-slate-200 dark:border-slate-700 flex items-start gap-4">
            <div className="p-3 bg-[#0056b2]/10 text-[#0056b2] rounded-lg">
              <span className="material-symbols-outlined">database</span>
            </div>
            <div>
              <h3 className="font-bold text-slate-900 dark:text-white">24 Clinical Features</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400">Comprehensive patient data integration</p>
            </div>
          </div>
          <div className="bg-white dark:bg-slate-800 p-5 rounded-xl border border-slate-200 dark:border-slate-700 flex items-start gap-4">
            <div className="p-3 bg-[#0056b2]/10 text-[#0056b2] rounded-lg">
              <span className="material-symbols-outlined">bolt</span>
            </div>
            <div>
              <h3 className="font-bold text-slate-900 dark:text-white">Real-time Feedback</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400">Instant risk probability calculations</p>
            </div>
          </div>
        </div>

        {/* Main Grid Content */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">

          {/* Left Column: Patient Profile */}
          <div className="lg:col-span-8">
            <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="px-6 py-4 border-b border-slate-100 dark:border-slate-700 flex justify-between items-center bg-slate-50/50 dark:bg-slate-800/50">
                <h2 className="text-xl font-bold text-slate-900 dark:text-white">Patient Profile</h2>
                <button onClick={fillSampleData} className="flex items-center gap-2 px-3 py-1.5 text-xs font-semibold bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-600 transition-all cursor-pointer">
                  <span className="material-symbols-outlined text-sm">content_paste_go</span>
                  Fill Sample Data
                </button>
              </div>

              <div className="p-6 space-y-8">

                {/* Demographics */}
                <section>
                  <h3 className="text-sm font-bold text-[#0056b2] uppercase tracking-widest mb-4 flex items-center gap-2">
                    <span className="material-symbols-outlined text-sm">person</span> Demographics
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-1.5 flex flex-col">
                      <label className="text-xs font-bold text-slate-500 uppercase">Age (Years)</label>
                      <input name="age" value={formData.age} onChange={handleChange} className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-2 focus:ring-[#0056b2] focus:border-[#0056b2] outline-none" placeholder="e.g. 48" type="number" />
                    </div>
                    <div className="space-y-1.5 flex flex-col">
                      <label className="text-xs font-bold text-slate-500 uppercase">Blood Pressure (mm/Hg)</label>
                      <input name="bp" value={formData.bp} onChange={handleChange} className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-2 focus:ring-[#0056b2] focus:border-[#0056b2] outline-none" placeholder="e.g. 80" type="number" />
                    </div>
                  </div>
                </section>

                {/* Urine Test Results */}
                <section>
                  <h3 className="text-sm font-bold text-[#0056b2] uppercase tracking-widest mb-4 flex items-center gap-2">
                    <span className="material-symbols-outlined text-sm">opacity</span> Urine Test Results
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="space-y-1.5 flex flex-col">
                      <label className="text-xs font-bold text-slate-500 uppercase">Specific Gravity</label>
                      <select name="sg" value={formData.sg} onChange={handleChange} className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-2 outline-none">
                        <option value="1.025">1.025</option>
                        <option value="1.020">1.020</option>
                        <option value="1.015">1.015</option>
                        <option value="1.010">1.010</option>
                        <option value="1.005">1.005</option>
                      </select>
                    </div>
                    <div className="space-y-1.5 flex flex-col">
                      <label className="text-xs font-bold text-slate-500 uppercase">Albumin</label>
                      <select name="al" value={formData.al} onChange={handleChange} className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-2 outline-none">
                        <option value="0">0 (Normal)</option>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="4">4+</option>
                        <option value="5">5+</option>
                      </select>
                    </div>
                    <div className="space-y-1.5 flex flex-col">
                      <label className="text-xs font-bold text-slate-500 uppercase">Sugar</label>
                      <select name="su" value={formData.su} onChange={handleChange} className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-2 outline-none">
                        <option value="0">0</option>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="4">4</option>
                        <option value="5">5+</option>
                      </select>
                    </div>
                    <div className="space-y-1.5 flex flex-col">
                      <label className="text-xs font-bold text-slate-500 uppercase">Red Blood Cells</label>
                      <select name="rbc" value={formData.rbc} onChange={handleChange} className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-2 outline-none">
                        <option value="normal">Normal</option>
                        <option value="abnormal">Abnormal</option>
                      </select>
                    </div>
                    <div className="space-y-1.5 flex flex-col">
                      <label className="text-xs font-bold text-slate-500 uppercase">Pus Cell / WBC</label>
                      <select name="pc" value={formData.pc} onChange={handleChange} className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-2 outline-none">
                        <option value="normal">Normal</option>
                        <option value="abnormal">Abnormal</option>
                      </select>
                    </div>
                    <div className="space-y-1.5 flex flex-col">
                      <label className="text-xs font-bold text-slate-500 uppercase">Pus Cell Clumps</label>
                      <select name="pcc" value={formData.pcc} onChange={handleChange} className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-2 outline-none">
                        <option value="notpresent">Not Present</option>
                        <option value="present">Present</option>
                      </select>
                    </div>
                    <div className="space-y-1.5 flex flex-col md:col-span-3">
                      <label className="text-xs font-bold text-slate-500 uppercase">Bacteria</label>
                      <select name="ba" value={formData.ba} onChange={handleChange} className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-2 outline-none">
                        <option value="notpresent">Not Present</option>
                        <option value="present">Present</option>
                      </select>
                    </div>
                  </div>
                </section>

                {/* Blood Test Results */}
                <section>
                  <h3 className="text-sm font-bold text-[#0056b2] uppercase tracking-widest mb-4 flex items-center gap-2">
                    <span className="material-symbols-outlined text-sm">bloodtype</span> Blood Test Results
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="space-y-1.5 flex flex-col">
                      <label className="text-xs font-bold text-slate-500 uppercase">Glucose (mg/dl)</label>
                      <input name="bgr" value={formData.bgr} onChange={handleChange} className="w-full rounded-lg border border-slate-200 bg-slate-50 dark:bg-slate-900 dark:border-slate-700 p-2 outline-none" type="number" />
                    </div>
                    <div className="space-y-1.5 flex flex-col">
                      <label className="text-xs font-bold text-slate-500 uppercase">Urea (mg/dl)</label>
                      <input name="bu" value={formData.bu} onChange={handleChange} className="w-full rounded-lg border border-slate-200 bg-slate-50 dark:bg-slate-900 dark:border-slate-700 p-2 outline-none" type="number" />
                    </div>
                    <div className="space-y-1.5 flex flex-col">
                      <label className="text-xs font-bold text-slate-500 uppercase">Creatinine (mg/dl)</label>
                      <input name="sc" value={formData.sc} onChange={handleChange} className="w-full rounded-lg border border-slate-200 bg-slate-50 dark:bg-slate-900 dark:border-slate-700 p-2 outline-none" type="number" step="0.1" />
                    </div>
                    <div className="space-y-1.5 flex flex-col">
                      <label className="text-xs font-bold text-slate-500 uppercase">Hemoglobin (g/dl)</label>
                      <input name="hemo" value={formData.hemo} onChange={handleChange} className="w-full rounded-lg border border-slate-200 bg-slate-50 dark:bg-slate-900 dark:border-slate-700 p-2 outline-none" type="number" step="0.1" />
                    </div>
                    <div className="space-y-1.5 flex flex-col">
                      <label className="text-xs font-bold text-slate-500 uppercase">Sodium (mEq/L)</label>
                      <input name="sod" value={formData.sod} onChange={handleChange} className="w-full rounded-lg border border-slate-200 bg-slate-50 dark:bg-slate-900 dark:border-slate-700 p-2 outline-none" type="number" />
                    </div>
                    <div className="space-y-1.5 flex flex-col">
                      <label className="text-xs font-bold text-slate-500 uppercase">Potassium (mEq/L)</label>
                      <input name="pot" value={formData.pot} onChange={handleChange} className="w-full rounded-lg border border-slate-200 bg-slate-50 dark:bg-slate-900 dark:border-slate-700 p-2 outline-none" type="number" step="0.1" />
                    </div>
                    <div className="space-y-1.5 flex flex-col">
                      <label className="text-xs font-bold text-slate-500 uppercase">PCV (%)</label>
                      <input name="pcv" value={formData.pcv} onChange={handleChange} className="w-full rounded-lg border border-slate-200 bg-slate-50 dark:bg-slate-900 dark:border-slate-700 p-2 outline-none" type="number" />
                    </div>
                    <div className="space-y-1.5 flex flex-col">
                      <label className="text-xs font-bold text-slate-500 uppercase">WBC Count</label>
                      <input name="wc" value={formData.wc} onChange={handleChange} className="w-full rounded-lg border border-slate-200 bg-slate-50 dark:bg-slate-900 dark:border-slate-700 p-2 outline-none" type="number" />
                    </div>
                    <div className="space-y-1.5 flex flex-col md:col-span-4">
                      <label className="text-xs font-bold text-slate-500 uppercase">RBC Count (millions/cumm)</label>
                      <input name="rc" value={formData.rc} onChange={handleChange} className="w-full rounded-lg border border-slate-200 bg-slate-50 dark:bg-slate-900 dark:border-slate-700 p-2 outline-none" type="number" step="0.1" />
                    </div>
                  </div>
                </section>

                {/* Medical History */}
                <section>
                  <h3 className="text-sm font-bold text-[#0056b2] uppercase tracking-widest mb-4 flex items-center gap-2">
                    <span className="material-symbols-outlined text-sm">history</span> Medical History
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                    <label className="flex items-center gap-3 cursor-pointer">
                      <input name="htn" checked={formData.htn === "yes"} onChange={handleChange} type="checkbox" className="rounded text-[#0056b2] focus:ring-[#0056b2] w-5 h-5 cursor-pointer" />
                      <span className="text-sm font-medium">Hypertension</span>
                    </label>
                    <label className="flex items-center gap-3 cursor-pointer">
                      <input name="dm" checked={formData.dm === "yes"} onChange={handleChange} type="checkbox" className="rounded text-[#0056b2] focus:ring-[#0056b2] w-5 h-5 cursor-pointer" />
                      <span className="text-sm font-medium">Diabetes</span>
                    </label>
                    <label className="flex items-center gap-3 cursor-pointer">
                      <input name="cad" checked={formData.cad === "yes"} onChange={handleChange} type="checkbox" className="rounded text-[#0056b2] focus:ring-[#0056b2] w-5 h-5 cursor-pointer" />
                      <span className="text-sm font-medium">CAD</span>
                    </label>
                    <label className="flex items-center gap-3 cursor-pointer">
                      <input name="appet" checked={formData.appet === "poor"} onChange={(e) => setFormData(p => ({ ...p, appet: e.target.checked ? "poor" : "good" }))} type="checkbox" className="rounded text-[#0056b2] focus:ring-[#0056b2] w-5 h-5 cursor-pointer" />
                      <span className="text-sm font-medium">Poor Appetite</span>
                    </label>
                    <label className="flex items-center gap-3 cursor-pointer">
                      <input name="pe" checked={formData.pe === "yes"} onChange={handleChange} type="checkbox" className="rounded text-[#0056b2] focus:ring-[#0056b2] w-5 h-5 cursor-pointer" />
                      <span className="text-sm font-medium">Pedal Edema</span>
                    </label>
                    <label className="flex items-center gap-3 cursor-pointer">
                      <input name="ane" checked={formData.ane === "yes"} onChange={handleChange} type="checkbox" className="rounded text-[#0056b2] focus:ring-[#0056b2] w-5 h-5 cursor-pointer" />
                      <span className="text-sm font-medium">Anemia</span>
                    </label>
                  </div>
                </section>

              </div>

              <div className="p-6 bg-slate-50 dark:bg-slate-800/80 border-t border-slate-100 dark:border-slate-700">
                <button
                  onClick={handlePredict}
                  disabled={isLoading || !health?.ok}
                  className="w-full bg-[#0056b2] disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#0056b2]/90 text-white font-bold py-4 rounded-xl shadow-lg shadow-[#0056b2]/20 flex items-center justify-center gap-3 transition-all transform active:scale-[0.98] cursor-pointer"
                >
                  {isLoading ? (
                    <div className="h-5 w-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  ) : (
                    <span className="material-symbols-outlined">analytics</span>
                  )}
                  {isLoading ? "Running Diagnostic Engine..." : "Predict CKD Risk"}
                </button>
                {error && <p className="text-red-500 text-sm mt-3 text-center font-medium bg-red-50 p-2 rounded">{error}</p>}
              </div>
            </div>
          </div>

          {/* Right Column: Diagnostic Output */}
          <div className="lg:col-span-4 space-y-6">
            <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden sticky top-24">
              <div className="px-6 py-4 border-b border-slate-100 dark:border-slate-700 bg-slate-50/50 dark:bg-slate-800/50">
                <h2 className="text-xl font-bold text-slate-900 dark:text-white">Diagnostic Output</h2>
              </div>

              <div className="p-6 space-y-6">
                {!prediction ? (
                  <div className="text-center py-12 px-4 border-2 border-dashed border-slate-200 dark:border-slate-700 rounded-xl">
                    <span className="material-symbols-outlined text-4xl text-slate-300 dark:text-slate-600 mb-2">biotech</span>
                    <h3 className="text-slate-500 dark:text-slate-400 font-medium">Awaiting Data</h3>
                    <p className="text-xs text-slate-400 mt-1">Input patient parameters and run the prediction engine to see results here.</p>
                  </div>
                ) : (
                  <>
                    {/* Risk Alert */}
                    <div className={`border p-5 rounded-xl text-center ${prediction.predicted_class === 1
                        ? "bg-red-50 dark:bg-red-950/30 border-red-100 dark:border-red-900/50 text-red-600 dark:text-red-400"
                        : "bg-emerald-50 dark:bg-emerald-950/30 border-emerald-100 dark:border-emerald-900/50 text-emerald-600 dark:text-emerald-400"
                      }`}>
                      <div className={`w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3 ${prediction.predicted_class === 1 ? "bg-red-100 dark:bg-red-900" : "bg-emerald-100 dark:bg-emerald-900"
                        }`}>
                        <span className="material-symbols-outlined text-3xl">
                          {prediction.predicted_class === 1 ? "warning" : "health_and_safety"}
                        </span>
                      </div>
                      <h3 className="text-lg font-bold">
                        {prediction.predicted_class === 1 ? "CKD Risk Detected" : "Low Risk (Healthy)"}
                      </h3>
                      <p className="text-sm opacity-80 mt-1">
                        {prediction.predicted_class === 1
                          ? "Immediate clinical follow-up recommended."
                          : "Parameters are within acceptable thresholds."}
                      </p>
                    </div>

                    {/* Confidence & Level */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-slate-50 dark:bg-slate-900/50 p-4 rounded-xl border border-slate-100 dark:border-slate-700 text-center">
                        <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Confidence Score</p>
                        <p className="text-2xl font-black text-[#0056b2]">{(prediction.confidence || 0).toFixed(1)}%</p>
                      </div>
                      <div className="bg-slate-50 dark:bg-slate-900/50 p-4 rounded-xl border border-slate-100 dark:border-slate-700 text-center">
                        <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Risk Assessment</p>
                        <p className={`text-2xl font-black ${prediction.predicted_class === 1 ? "text-red-600" : "text-emerald-600"}`}>
                          {prediction.predicted_class === 1 ? "High" : "Low"}
                        </p>
                      </div>
                    </div>

                    {/* Probability Breakdown */}
                    <div className="space-y-4">
                      <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Probability Breakdown</h4>
                      <div className="space-y-3">
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="font-medium">CKD Positive</span>
                            <span className="font-bold">{(prediction.probabilities?.ckd || 0).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-slate-200 dark:bg-slate-700 h-2 rounded-full overflow-hidden">
                            <div className="bg-red-500 h-full rounded-full transition-all duration-1000" style={{ width: `${prediction.probabilities?.ckd || 0}%` }}></div>
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between text-sm mb-1 text-slate-400">
                            <span className="font-medium">No CKD Risk</span>
                            <span className="font-bold">{(prediction.probabilities?.not_ckd || 0).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-slate-200 dark:bg-slate-700 h-2 rounded-full overflow-hidden">
                            <div className="bg-emerald-400 h-full rounded-full transition-all duration-1000" style={{ width: `${prediction.probabilities?.not_ckd || 0}%` }}></div>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="pt-6 border-t border-slate-100 dark:border-slate-700 space-y-4">
                      <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
                        <span className="font-bold">Model Engine</span>
                        <span>{prediction.model || 'Machine Learning API'}</span>
                      </div>
                      <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
                        <span className="font-bold">Analysis Time</span>
                        <span>{new Date(prediction.timestamp).toLocaleTimeString() || "Just now"}</span>
                      </div>
                    </div>
                  </>
                )}

                {/* Disclaimer */}
                <div className="bg-[#0056b2]/5 rounded-lg p-4 flex gap-3 items-start border border-[#0056b2]/10 mt-4">
                  <span className="material-symbols-outlined text-[#0056b2] text-lg shrink-0">info</span>
                  <p className="text-[11px] leading-relaxed text-slate-600 dark:text-slate-400">
                    <span className="font-bold text-[#0056b2]">Disclaimer:</span> This AI tool is for preliminary screening and educational purposes only. Results must be verified by a certified medical professional and are not a final diagnosis.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="max-w-7xl mx-auto px-6 py-10 mt-8 border-t border-slate-200 dark:border-slate-800 text-center">
        <p className="text-xs text-slate-500 dark:text-slate-400">
          © {new Date().getFullYear()} CKD Prediction. Powered by Advanced Clinical Analytics.
        </p>
      </footer>
    </div>
  );
};

export default Home;
