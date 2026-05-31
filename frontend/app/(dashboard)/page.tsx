/**
 * Dashboard Home Page — Premium Glass Design
 * ============================================
 * Gradient background, glass stat cards, quota bar, recent predictions.
 * Uses btn-primary/btn-secondary for 3-color button rule.
 */

"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { apiService } from "@/lib/api";
import { useAuthStore } from "@/lib/auth";
import type { AnalyticsSummary, PredictionRecord } from "@/lib/types";
import { ProgressBar } from "@/components/ui/progress-bar";
import { Badge } from "@/components/ui/badge";
import {
  BarChart3,
  Clock,
  Image,
  Zap,
  ArrowRight,
  TrendingUp,
  Activity,
  Sparkles,
} from "lucide-react";

interface GlassStatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  trend?: string;
  accent: string; // tailwind gradient class
}

function GlassStatCard({ title, value, subtitle, icon, trend, accent }: GlassStatCardProps) {
  return (
    <div className="glass rounded-2xl p-5 shadow-glass dark:shadow-glass-dark hover:shadow-glass-lg transition-all duration-300 border border-white/20 dark:border-white/5">
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</p>
        <div className={`p-2.5 rounded-xl ${accent}`}>
          {icon}
        </div>
      </div>
      <p className="text-3xl font-bold text-gray-900 dark:text-gray-100">{value}</p>
      <div className="flex items-center gap-2 mt-1">
        {subtitle && <p className="text-sm text-gray-400 dark:text-gray-500">{subtitle}</p>}
        {trend && (
          <span className="flex items-center gap-1 text-xs text-primary-500 dark:text-primary-400 bg-primary-400/10 px-2 py-0.5 rounded-full border border-primary-400/20">
            <TrendingUp className="w-3 h-3" /> {trend}
          </span>
        )}
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const user = useAuthStore((s) => s.user);
  const tenant = useAuthStore((s) => s.tenant);
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [recentPredictions, setRecentPredictions] = useState<PredictionRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [summaryData, historyData] = await Promise.all([
          apiService.getAnalyticsSummary(),
          apiService.getHistory({ page: 1, page_size: 5 }),
        ]);
        setSummary(summaryData);
        setRecentPredictions(historyData.items);
      } catch (error) {
        console.error("Failed to load dashboard data:", error);
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

  const quotaLimit = tenant?.plan === "pro" ? 1000 : tenant?.plan === "enterprise" ? 10000 : 100;
  const quotaUsed = summary?.quota_used ?? 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Welcome back{user?.display_name ? `, ${user.display_name}` : ""}
          </h1>
          <p className="text-gray-400 dark:text-gray-500 mt-1 flex items-center gap-2">
            {tenant?.name || "Your workspace"} — <Badge variant={tenant?.plan === "pro" ? "pro" : tenant?.plan === "enterprise" ? "enterprise" : "free"}>{tenant?.plan || "free"}</Badge>
          </p>
        </div>
        <Link
          href="/predict"
          className="btn-primary shadow-glow"
        >
          <Sparkles className="w-4 h-4" /> New Prediction
        </Link>
      </div>

      {/* Glass Stat Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <GlassStatCard
          title="Total Predictions"
          value={summary?.total_predictions ?? 0}
          icon={<Image className="w-5 h-5 text-white" />}
          accent="bg-gradient-to-br from-primary-400 to-primary-600"
        />
        <GlassStatCard
          title="This Month"
          value={summary?.predictions_this_month ?? 0}
          icon={<BarChart3 className="w-5 h-5 text-white" />}
          accent="bg-gradient-to-br from-emerald-400 to-emerald-600"
          trend="Active"
        />
        <GlassStatCard
          title="Avg. Processing"
          value={summary?.avg_processing_ms ? `${Math.round(summary.avg_processing_ms)}ms` : "—"}
          icon={<Clock className="w-5 h-5 text-white" />}
          accent="bg-gradient-to-br from-violet-400 to-violet-600"
        />
        <GlassStatCard
          title="Quota Used"
          value={`${quotaUsed} / ${quotaLimit}`}
          icon={<Zap className="w-5 h-5 text-white" />}
          accent="bg-gradient-to-br from-amber-400 to-amber-600"
        />
      </div>

      {/* Quota Progress */}
      <div className="glass rounded-2xl p-6 shadow-glass dark:shadow-glass-dark border border-white/20 dark:border-white/5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
            <Activity className="w-5 h-5 text-primary-400" /> Monthly Quota
          </h2>
          <Link href="/settings" className="text-sm text-primary-400 hover:text-primary-500 dark:text-primary-400 transition-colors">
            Manage Plan
          </Link>
        </div>
        <ProgressBar value={quotaUsed} max={quotaLimit} label="Predictions" />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Link
          href="/predict"
          className="glass flex items-center gap-3 p-4 rounded-2xl shadow-glass-sm dark:shadow-glass-dark hover:shadow-glass-lg transition-all duration-300 group border border-white/20 dark:border-white/5 hover:border-primary-400/30"
        >
          <div className="p-2.5 bg-primary-400/10 dark:bg-primary-400/15 rounded-xl group-hover:bg-primary-400/20 transition-colors">
            <Image className="w-5 h-5 text-primary-400" />
          </div>
          <div className="flex-1">
            <p className="font-medium text-gray-900 dark:text-gray-100">New Prediction</p>
            <p className="text-sm text-gray-400 dark:text-gray-500">Upload and classify images</p>
          </div>
          <ArrowRight className="w-4 h-4 text-gray-300 dark:text-gray-600 group-hover:text-primary-400 transition-colors" />
        </Link>

        <Link
          href="/api-keys"
          className="glass flex items-center gap-3 p-4 rounded-2xl shadow-glass-sm dark:shadow-glass-dark hover:shadow-glass-lg transition-all duration-300 group border border-white/20 dark:border-white/5 hover:border-emerald-400/30"
        >
          <div className="p-2.5 bg-emerald-400/10 dark:bg-emerald-400/15 rounded-xl group-hover:bg-emerald-400/20 transition-colors">
            <Zap className="w-5 h-5 text-emerald-500" />
          </div>
          <div className="flex-1">
            <p className="font-medium text-gray-900 dark:text-gray-100">API Keys</p>
            <p className="text-sm text-gray-400 dark:text-gray-500">Manage programmatic access</p>
          </div>
          <ArrowRight className="w-4 h-4 text-gray-300 dark:text-gray-600 group-hover:text-emerald-400 transition-colors" />
        </Link>

        <Link
          href="/analytics"
          className="glass flex items-center gap-3 p-4 rounded-2xl shadow-glass-sm dark:shadow-glass-dark hover:shadow-glass-lg transition-all duration-300 group border border-white/20 dark:border-white/5 hover:border-violet-400/30"
        >
          <div className="p-2.5 bg-violet-400/10 dark:bg-violet-400/15 rounded-xl group-hover:bg-violet-400/20 transition-colors">
            <BarChart3 className="w-5 h-5 text-violet-500" />
          </div>
          <div className="flex-1">
            <p className="font-medium text-gray-900 dark:text-gray-100">Analytics</p>
            <p className="text-sm text-gray-400 dark:text-gray-500">View prediction distributions</p>
          </div>
          <ArrowRight className="w-4 h-4 text-gray-300 dark:text-gray-600 group-hover:text-violet-400 transition-colors" />
        </Link>
      </div>

      {/* Recent Predictions */}
      <div className="glass rounded-2xl shadow-glass dark:shadow-glass-dark border border-white/20 dark:border-white/5">
        <div className="px-6 py-4 border-b border-white/10 dark:border-white/5 flex items-center justify-between">
          <h2 className="font-semibold text-gray-900 dark:text-gray-100">Recent Predictions</h2>
          <Link
            href="/history"
            className="text-sm text-primary-400 hover:text-primary-500 dark:text-primary-400 transition-colors flex items-center gap-1"
          >
            View all <ArrowRight className="w-3 h-3" />
          </Link>
        </div>
        {recentPredictions.length === 0 ? (
          <div className="px-6 py-16 text-center">
            <div className="w-16 h-16 glass rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-glass-sm border border-white/20 dark:border-white/5">
              <Image className="w-8 h-8 text-gray-300 dark:text-gray-600" />
            </div>
            <p className="text-gray-500 dark:text-gray-400 font-medium">No predictions yet</p>
            <p className="text-sm text-gray-400 dark:text-gray-500 mt-1">
              Upload your first image to get started
            </p>
            <Link
              href="/predict"
              className="btn-primary shadow-glow mt-4"
            >
              <Sparkles className="w-4 h-4" /> Make Prediction
            </Link>
          </div>
        ) : (
          <div className="divide-y divide-white/5 dark:divide-white/5">
            {recentPredictions.map((pred) => (
              <Link
                key={pred.id}
                href={`/history/${pred.id}`}
                className="flex items-center justify-between px-6 py-3.5 hover:bg-black/[0.02] dark:hover:bg-white/[0.02] transition-all duration-200 group"
              >
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 glass rounded-xl flex items-center justify-center border border-white/20 dark:border-white/5 group-hover:border-primary-400/30 transition-colors">
                    <Image className="w-4 h-4 text-gray-400 dark:text-gray-500" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      {pred.filename || "Untitled"}
                    </p>
                    <p className="text-xs text-gray-400 dark:text-gray-500">
                      {pred.theme && `${pred.theme} • `}
                      {pred.sentiment || "—"}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-xs text-gray-400 dark:text-gray-500">
                    {pred.processing_ms ? `${pred.processing_ms}ms` : "—"}
                  </p>
                  <p className="text-xs text-gray-400 dark:text-gray-600">
                    {new Date(pred.created_at).toLocaleDateString()}
                  </p>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
