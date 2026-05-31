/**
 * Prediction Detail Page
 * ======================
 * Full detail view of a single prediction.
 */

"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { apiService } from "@/lib/api";
import type { PredictionDetail } from "@/lib/types";
import { ArrowLeft, Clock, FileText, Trash2, AlertCircle } from "lucide-react";

export default function PredictionDetailPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const [prediction, setPrediction] = useState<PredictionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const data = await apiService.getPrediction(id);
        setPrediction(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load prediction");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [id]);

  const handleDelete = async () => {
    if (!confirm("Delete this prediction?")) return;
    try {
      await apiService.deletePrediction(id);
      router.push("/history");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete");
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-400" />
      </div>
    );
  }

  if (error || !prediction) {
    return (
      <div className="space-y-4">
        <button
          onClick={() => router.push("/history")}
          className="flex items-center gap-2 text-gray-600 hover:text-gray-900"
        >
          <ArrowLeft className="w-4 h-4" /> Back to History
        </button>
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2 text-red-700">
          <AlertCircle className="w-4 h-4" />
          {error || "Prediction not found"}
        </div>
      </div>
    );
  }

  const result = prediction.result as Record<string, string>;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => router.push("/history")}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </button>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              {prediction.filename || "Untitled"}
            </h1>
            <p className="text-sm text-gray-500 mt-1">
              {new Date(prediction.created_at).toLocaleString()}
            </p>
          </div>
        </div>
        <button
          onClick={handleDelete}
          className="flex items-center gap-2 px-3 py-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
        >
          <Trash2 className="w-4 h-4" /> Delete
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <p className="text-sm text-gray-500">Processing Time</p>
          <p className="text-lg font-semibold text-gray-900">
            {prediction.processing_ms ? `${prediction.processing_ms}ms` : "—"}
          </p>
        </div>
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <p className="text-sm text-gray-500">Predicted Label</p>
          <p className="text-lg font-semibold text-gray-900">
            {result.predicted_label || "—"}
          </p>
        </div>
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <p className="text-sm text-gray-500">Theme</p>
          <p className="text-lg font-semibold text-gray-900">{result.theme || "—"}</p>
        </div>
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <p className="text-sm text-gray-500">Sentiment</p>
          <p className="text-lg font-semibold text-gray-900">{result.sentiment || "—"}</p>
        </div>
      </div>

      {/* All Attributes */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="font-semibold text-gray-900">All Attributes</h2>
        </div>
        <div className="divide-y divide-gray-100">
          {[
            "theme",
            "sentiment",
            "emotion",
            "dominant_colour",
            "attention_score",
            "trust_safety",
            "target_audience",
            "predicted_ctr",
            "likelihood_shares",
          ].map((attr) => (
            <div key={attr} className="px-6 py-3 flex items-center justify-between">
              <span className="text-sm text-gray-500 capitalize">
                {attr.replace(/_/g, " ")}
              </span>
              <span className="text-sm font-medium text-gray-900">
                {result[attr] || "—"}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* OCR Text */}
      {prediction.ocr_text && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="px-6 py-4 border-b border-gray-200 flex items-center gap-2">
            <FileText className="w-4 h-4 text-gray-400" />
            <h2 className="font-semibold text-gray-900">OCR Text</h2>
          </div>
          <div className="px-6 py-4">
            <p className="text-sm text-gray-700 whitespace-pre-wrap">{prediction.ocr_text}</p>
          </div>
        </div>
      )}

      {/* Extracted Features */}
      {(["keywords", "monetary_mention", "call_to_action", "object_detected"] as const).some(
        (k) => result[k]
      ) && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="font-semibold text-gray-900">Extracted Features</h2>
          </div>
          <div className="divide-y divide-gray-100">
            {(["keywords", "monetary_mention", "call_to_action", "object_detected"] as const).map(
              (feature) =>
                result[feature] ? (
                  <div key={feature} className="px-6 py-3 flex items-center justify-between">
                    <span className="text-sm text-gray-500 capitalize">
                      {feature.replace(/_/g, " ")}
                    </span>
                    <span className="text-sm font-medium text-gray-900">{result[feature]}</span>
                  </div>
                ) : null
            )}
          </div>
        </div>
      )}
    </div>
  );
}
