/**
 * Badge Component — Premium Design
 * ==================================
 * Uses #ff6b35 for pro variant. Glass effect + border.
 */

"use client";

interface BadgeProps {
  variant?: "free" | "pro" | "enterprise" | "default" | "success" | "warning" | "danger";
  children: React.ReactNode;
}

const VARIANT_CLASSES: Record<string, string> = {
  free: "bg-gray-100/80 text-gray-600 border-gray-200 dark:bg-gray-800/80 dark:text-gray-300 dark:border-gray-700",
  pro: "bg-primary-400/10 text-primary-500 border-primary-400/20 dark:bg-primary-400/15 dark:text-primary-400 dark:border-primary-400/20",
  enterprise: "bg-violet-100/80 text-violet-700 border-violet-200 dark:bg-violet-900/30 dark:text-violet-300 dark:border-violet-700/30",
  default: "bg-gray-100/80 text-gray-600 border-gray-200 dark:bg-gray-800/80 dark:text-gray-300 dark:border-gray-700",
  success: "bg-emerald-100/80 text-emerald-700 border-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-300 dark:border-emerald-700/30",
  warning: "bg-amber-100/80 text-amber-700 border-amber-200 dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-700/30",
  danger: "bg-red-100/80 text-red-700 border-red-200 dark:bg-red-900/30 dark:text-red-300 dark:border-red-700/30",
};

export function Badge({ variant = "default", children }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold capitalize backdrop-blur-sm border transition-all duration-200 ${VARIANT_CLASSES[variant]}`}
    >
      {children}
    </span>
  );
}
