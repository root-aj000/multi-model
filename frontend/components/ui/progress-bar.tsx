/**
 * Progress Bar Component
 * ======================
 * Quota usage progress bar with color coding.
 */

"use client";

interface ProgressBarProps {
  value: number;
  max: number;
  label?: string;
  showText?: boolean;
}

export function ProgressBar({ value, max, label, showText = true }: ProgressBarProps) {
  const percentage = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  const isWarning = percentage >= 80;
  const isDanger = percentage >= 95;

  const barColor = isDanger
    ? "bg-red-500"
    : isWarning
    ? "bg-amber-500"
    : "bg-primary-400/50";

  return (
    <div>
      {(label || showText) && (
        <div className="flex items-center justify-between mb-1">
          {label && (
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{label}</span>
          )}
          {showText && (
            <span className={`text-sm font-medium ${
              isDanger ? "text-red-600 dark:text-red-400" : isWarning ? "text-amber-600 dark:text-amber-400" : "text-gray-500 dark:text-gray-400"
            }`}>
              {value} / {max}
            </span>
          )}
        </div>
      )}
      <div className="w-full h-2.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${barColor}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      {isDanger && (
        <p className="text-xs text-red-600 dark:text-red-400 mt-1">Quota almost exhausted — consider upgrading your plan</p>
      )}
    </div>
  );
}
