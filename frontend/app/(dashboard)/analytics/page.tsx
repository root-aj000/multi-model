/**
 * Analytics Page
 * =============
 * Attribute distribution charts and summary statistics.
 */

"use client";

import { useEffect, useState } from "react";
import { apiService } from "@/lib/api";
import type { AnalyticsSummary, AttributeDistributions } from "@/lib/types";
import { StatCard } from "@/components/dashboard/stat-card";
import {
  BarChart3,
  Clock,
  Image,
  Zap,
} from "lucide-react";

// Simple bar chart component (no recharts dependency needed for prototype)
function SimpleBarChart({ data, title }: { data: Record<string, number>; title: string }) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);
  const maxVal = Math.max(...entries.map(([, v]) => v), 1);

  if (entries.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-900 mb-4 capitalize">{title}</h3>
        <p className="text-sm text-gray-400">No data available</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h3 className="font-semibold text-gray-900 mb-4 capitalize">
        {title.replace(/_/g, " ")}
      </h3>
      <div className="space-y-2">
        {entries.map(([label, count]) => (
          <div key={label} className="flex items-center gap-3">
            <span className="text-sm text-gray-600 w-28 truncate" title={label}>
              {label}
            </span>
            <div className="flex-1 bg-gray-100 rounded-full h-6 relative overflow-hidden">
              <div
                className="bg-primary-400/50 h-full rounded-full transition-all duration-500"
                style={{ width: `${(count / maxVal) * 100}%` }}
              />
            </div>
            <span className="text-sm font-medium text-gray-900 w-10 text-right">{count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function AnalyticsPage() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [distributions, setDistributions] = useState<AttributeDistributions | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [summaryData, distData] = await Promise.all([
          apiService.getAnalyticsSummary(),
          apiService.getAnalyticsAttributes(),
        ]);
        setSummary(summaryData);
        setDistributions(distData);
      } catch (error) {
        console.error("Failed to load analytics:", error);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-400" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
        <p className="text-gray-500 mt-1">Prediction attribute distributions and trends</p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Predictions"
          value={summary?.total_predictions ?? 0}
          icon={<Image className="w-5 h-5" />}
          color="blue"
        />
        <StatCard
          title="This Month"
          value={summary?.predictions_this_month ?? 0}
          icon={<BarChart3 className="w-5 h-5" />}
          color="green"
        />
        <StatCard
          title="Avg. Processing"
          value={summary?.avg_processing_ms ? `${Math.round(summary.avg_processing_ms)}ms` : "—"}
          icon={<Clock className="w-5 h-5" />}
          color="purple"
        />
        <StatCard
          title="Quota Used"
          value={`${summary?.quota_used ?? 0} / ${summary?.quota_limit ?? 100}`}
          icon={<Zap className="w-5 h-5" />}
          color="amber"
        />
      </div>

      {/* Attribute Distributions */}
      {distributions && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <SimpleBarChart data={distributions.theme} title="Theme" />
          <SimpleBarChart data={distributions.sentiment} title="Sentiment" />
          <SimpleBarChart data={distributions.emotion} title="Emotion" />
          <SimpleBarChart data={distributions.dominant_colour} title="Dominant Colour" />
          <SimpleBarChart data={distributions.attention_score} title="Attention Score" />
          <SimpleBarChart data={distributions.trust_safety} title="Trust & Safety" />
          <SimpleBarChart data={distributions.target_audience} title="Target Audience" />
          <SimpleBarChart data={distributions.predicted_ctr} title="Predicted CTR" />
          <SimpleBarChart data={distributions.likelihood_shares} title="Likelihood Shares" />
        </div>
      )}
    </div>
  );
}
