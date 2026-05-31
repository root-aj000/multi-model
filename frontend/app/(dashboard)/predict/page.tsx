/**
 * Predict Page
 * ============
 * Migrated from the original single-page app.
 * Upload images and get multi-attribute predictions.
 */

"use client";

import { useState, useCallback, useEffect } from "react";
import ImageUpload from "@/components/ImageUpload";
import PredictionResults from "@/components/PredictionResults";
import { apiService } from "@/lib/api";
import type { PredictionResult, UploadFile } from "@/lib/types";
import { AlertCircle, Loader2 } from "lucide-react";

export default function PredictPage() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadFile[]>([]);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");

  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      await apiService.healthCheck();
      setApiStatus("online");
    } catch {
      setApiStatus("offline");
    }
  };

  const handleFilesSelected = useCallback((files: UploadFile[]) => {
    setUploadedFiles(files);
    setPredictions([]);
    setError(null);
    setProcessingTime(null);
  }, []);

  const handlePredict = async () => {
    if (uploadedFiles.length === 0) return;

    setIsLoading(true);
    setError(null);

    try {
      const files = uploadedFiles.map((f) => f.file);
      const result = await apiService.predict(files);
      setPredictions(result.predictions);
      setProcessingTime(result.processing_time_ms ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setUploadedFiles([]);
    setPredictions([]);
    setError(null);
    setProcessingTime(null);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Predict</h1>
        <p className="text-gray-500 mt-1">Upload images to classify with 9 attributes</p>
      </div>

      {/* API Status */}
      {apiStatus === "offline" && (
        <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg flex items-center gap-2 text-sm text-amber-700">
          <AlertCircle className="w-4 h-4 shrink-0" />
          API server is offline. Please check if the backend is running.
        </div>
      )}

      {/* Upload Section */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <ImageUpload onFilesSelected={handleFilesSelected} />

        <div className="mt-4 flex items-center gap-3">
          <button
            onClick={handlePredict}
            disabled={isLoading || uploadedFiles.length === 0}
            className="flex items-center gap-2 px-6 py-2 bg-primary-400 text-white rounded-lg hover:bg-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Processing...
              </>
            ) : (
              "Classify Images"
            )}
          </button>

          {uploadedFiles.length > 0 && !isLoading && (
            <button
              onClick={handleClear}
              className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2 text-sm text-red-700">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {/* Results */}
      {predictions.length > 0 && (
        <div id="results-section">
          <PredictionResults predictions={predictions} uploadedFiles={uploadedFiles} />
        </div>
      )}
    </div>
  );
}
