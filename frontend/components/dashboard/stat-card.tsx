/**
 * Stat Card Component
 * ===================
 * Displays a single metric with icon, value, and label.
 */

"use client";

import type { ReactNode } from "react";

interface StatCardProps {
  title: string;
  value: string | number;
  icon: ReactNode;
  color?: "blue" | "green" | "purple" | "amber" | "red";
  subtitle?: string;
}

const COLOR_MAP = {
  blue: { bg: "bg-primary-400/10", text: "text-primary-400" },
  green: { bg: "bg-green-100", text: "text-green-600" },
  purple: { bg: "bg-purple-100", text: "text-purple-600" },
  amber: { bg: "bg-amber-100", text: "text-amber-600" },
  red: { bg: "bg-red-100", text: "text-red-600" },
};

export function StatCard({ title, value, icon, color = "blue", subtitle }: StatCardProps) {
  const colors = COLOR_MAP[color];

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-500">{title}</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
          {subtitle && <p className="text-xs text-gray-400 mt-1">{subtitle}</p>}
        </div>
        <div className={`p-3 ${colors.bg} rounded-lg`}>
          <div className={colors.text}>{icon}</div>
        </div>
      </div>
    </div>
  );
}
