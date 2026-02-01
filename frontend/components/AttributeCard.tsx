'use client';

/**
 * Attribute Card Component
 * ========================
 * Displays a single attribute with its value.
 * Used to show prediction results in a visually appealing way.
 * 
 * Features:
 * - Icon support
 * - Color theming
 * - Muted state for missing values
 * - Hover effects
 * - Responsive design
 */

import { LucideIcon } from 'lucide-react';

interface AttributeCardProps {
  label: string;
  value: string;
  icon: LucideIcon;
  color: string;
  muted?: boolean;
}

/**
 * Color mapping for different attribute types.
 * Each color has background, text, and icon variants.
 */
const colorClasses: Record<string, {
  bg: string;
  text: string;
  icon: string;
  border: string;
  hoverBg: string;
}> = {
  blue: {
    bg: 'bg-blue-50',
    text: 'text-blue-700',
    icon: 'text-blue-500',
    border: 'border-blue-200',
    hoverBg: 'hover:bg-blue-100',
  },
  pink: {
    bg: 'bg-pink-50',
    text: 'text-pink-700',
    icon: 'text-pink-500',
    border: 'border-pink-200',
    hoverBg: 'hover:bg-pink-100',
  },
  purple: {
    bg: 'bg-purple-50',
    text: 'text-purple-700',
    icon: 'text-purple-500',
    border: 'border-purple-200',
    hoverBg: 'hover:bg-purple-100',
  },
  orange: {
    bg: 'bg-orange-50',
    text: 'text-orange-700',
    icon: 'text-orange-500',
    border: 'border-orange-200',
    hoverBg: 'hover:bg-orange-100',
  },
  yellow: {
    bg: 'bg-yellow-50',
    text: 'text-yellow-700',
    icon: 'text-yellow-500',
    border: 'border-yellow-200',
    hoverBg: 'hover:bg-yellow-100',
  },
  green: {
    bg: 'bg-green-50',
    text: 'text-green-700',
    icon: 'text-green-500',
    border: 'border-green-200',
    hoverBg: 'hover:bg-green-100',
  },
  cyan: {
    bg: 'bg-cyan-50',
    text: 'text-cyan-700',
    icon: 'text-cyan-500',
    border: 'border-cyan-200',
    hoverBg: 'hover:bg-cyan-100',
  },
  indigo: {
    bg: 'bg-indigo-50',
    text: 'text-indigo-700',
    icon: 'text-indigo-500',
    border: 'border-indigo-200',
    hoverBg: 'hover:bg-indigo-100',
  },
  emerald: {
    bg: 'bg-emerald-50',
    text: 'text-emerald-700',
    icon: 'text-emerald-500',
    border: 'border-emerald-200',
    hoverBg: 'hover:bg-emerald-100',
  },
  gray: {
    bg: 'bg-gray-50',
    text: 'text-gray-700',
    icon: 'text-gray-500',
    border: 'border-gray-200',
    hoverBg: 'hover:bg-gray-100',
  },
  red: {
    bg: 'bg-red-50',
    text: 'text-red-700',
    icon: 'text-red-500',
    border: 'border-red-200',
    hoverBg: 'hover:bg-red-100',
  },
};

export default function AttributeCard({
  label,
  value,
  icon: Icon,
  color,
  muted = false,
}: AttributeCardProps) {
  // Get color classes or default to gray
  const colors = colorClasses[color] || colorClasses.gray;

  // If muted, use gray colors
  const activeColors = muted ? colorClasses.gray : colors;

  return (
    <div
      className={`
        flex items-start gap-3 p-4 rounded-lg border transition-colors
        ${activeColors.bg}
        ${activeColors.border}
        ${activeColors.hoverBg}
        ${muted ? 'opacity-60' : ''}
      `}
    >
      {/* Icon */}
      <div
        className={`
          shrink-0 w-10 h-10 rounded-lg flex items-center justify-center
          ${muted ? 'bg-gray-100' : `bg-white shadow-sm`}
        `}
      >
        <Icon className={`w-5 h-5 ${activeColors.icon}`} />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        {/* Label */}
        <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">
          {label}
        </p>

        {/* Value */}
        <p
          className={`
            mt-1 text-base font-semibold truncate
            ${muted ? 'text-gray-400 italic' : activeColors.text}
          `}
          title={value}
        >
          {value}
        </p>
      </div>
    </div>
  );
}

/**
 * Compact version of AttributeCard for smaller displays.
 */
export function AttributeCardCompact({
  label,
  value,
  icon: Icon,
  color,
  muted = false,
}: AttributeCardProps) {
  // Get color classes or default to gray
  const colors = colorClasses[color] || colorClasses.gray;

  // If muted, use gray colors
  const activeColors = muted ? colorClasses.gray : colors;

  return (
    <div
      className={`
        flex items-center gap-2 px-3 py-2 rounded-lg border
        ${activeColors.bg}
        ${activeColors.border}
        ${muted ? 'opacity-60' : ''}
      `}
    >
      {/* Icon */}
      <Icon className={`w-4 h-4 shrink-0 ${activeColors.icon}`} />

      {/* Label */}
      <span className="text-xs text-gray-600">{label}:</span>

      {/* Value */}
      <span
        className={`
          text-sm font-medium truncate
          ${muted ? 'text-gray-400 italic' : activeColors.text}
        `}
        title={value}
      >
        {value}
      </span>
    </div>
  );
}

/**
 * Badge version of AttributeCard for inline display.
 */
export function AttributeBadge({
  label,
  value,
  color,
  muted = false,
}: Omit<AttributeCardProps, 'icon'>) {
  // Get color classes or default to gray
  const colors = colorClasses[color] || colorClasses.gray;

  // If muted, use gray colors
  const activeColors = muted ? colorClasses.gray : colors;

  return (
    <span
      className={`
        inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium
        ${activeColors.bg}
        ${activeColors.text}
        ${muted ? 'opacity-60' : ''}
      `}
      title={`${label}: ${value}`}
    >
      <span className="text-gray-500">{label}:</span>
      <span>{value}</span>
    </span>
  );
}