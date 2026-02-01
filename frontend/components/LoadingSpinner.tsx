'use client';

/**
 * Loading Spinner Component
 * =========================
 * Animated loading indicator for async operations.
 * 
 * Features:
 * - Multiple size variants
 * - Customizable colors
 * - Smooth animation
 * - Accessible with aria labels
 * - Optional loading text
 */

import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  color?: 'blue' | 'gray' | 'white' | 'green' | 'red';
  text?: string;
  fullScreen?: boolean;
  className?: string;
}

/**
 * Size classes for the spinner.
 */
const sizeClasses = {
  sm: 'w-4 h-4',
  md: 'w-8 h-8',
  lg: 'w-12 h-12',
  xl: 'w-16 h-16',
};

/**
 * Text size classes corresponding to spinner size.
 */
const textSizeClasses = {
  sm: 'text-xs',
  md: 'text-sm',
  lg: 'text-base',
  xl: 'text-lg',
};

/**
 * Color classes for the spinner.
 */
const colorClasses = {
  blue: 'text-blue-600',
  gray: 'text-gray-600',
  white: 'text-white',
  green: 'text-green-600',
  red: 'text-red-600',
};

export default function LoadingSpinner({
  size = 'lg',
  color = 'blue',
  text,
  fullScreen = false,
  className = '',
}: LoadingSpinnerProps) {
  // Container for full screen mode
  if (fullScreen) {
    return (
      <div
        className="fixed inset-0 bg-white/80 backdrop-blur-sm flex flex-col items-center justify-center z-50"
        role="status"
        aria-label="Loading"
      >
        <Loader2
          className={`
            animate-spin
            ${sizeClasses[size]}
            ${colorClasses[color]}
          `}
        />
        {text && (
          <p
            className={`
              mt-4 font-medium text-gray-700
              ${textSizeClasses[size]}
            `}
          >
            {text}
          </p>
        )}
        <span className="sr-only">Loading...</span>
      </div>
    );
  }

  // Inline spinner
  return (
    <div
      className={`flex flex-col items-center justify-center ${className}`}
      role="status"
      aria-label="Loading"
    >
      <Loader2
        className={`
          animate-spin
          ${sizeClasses[size]}
          ${colorClasses[color]}
        `}
      />
      {text && (
        <p
          className={`
            mt-3 font-medium text-gray-600
            ${textSizeClasses[size]}
          `}
        >
          {text}
        </p>
      )}
      <span className="sr-only">Loading...</span>
    </div>
  );
}

/**
 * Simple inline spinner for buttons.
 */
export function ButtonSpinner({
  size = 'sm',
  color = 'white',
}: Pick<LoadingSpinnerProps, 'size' | 'color'>) {
  return (
    <Loader2
      className={`
        animate-spin inline-block
        ${sizeClasses[size]}
        ${colorClasses[color]}
      `}
      aria-hidden="true"
    />
  );
}

/**
 * Skeleton loader for content placeholders.
 */
export function Skeleton({
  width = 'full',
  height = '4',
  rounded = 'md',
  className = '',
}: {
  width?: 'full' | '3/4' | '1/2' | '1/4';
  height?: '2' | '4' | '6' | '8' | '10' | '12' | '16' | '20' | '24' | '32';
  rounded?: 'none' | 'sm' | 'md' | 'lg' | 'full';
  className?: string;
}) {
  const widthClasses = {
    full: 'w-full',
    '3/4': 'w-3/4',
    '1/2': 'w-1/2',
    '1/4': 'w-1/4',
  };

  const heightClasses = {
    '2': 'h-2',
    '4': 'h-4',
    '6': 'h-6',
    '8': 'h-8',
    '10': 'h-10',
    '12': 'h-12',
    '16': 'h-16',
    '20': 'h-20',
    '24': 'h-24',
    '32': 'h-32',
  };

  const roundedClasses = {
    none: 'rounded-none',
    sm: 'rounded-sm',
    md: 'rounded-md',
    lg: 'rounded-lg',
    full: 'rounded-full',
  };

  return (
    <div
      className={`
        animate-pulse bg-gray-200
        ${widthClasses[width]}
        ${heightClasses[height]}
        ${roundedClasses[rounded]}
        ${className}
      `}
      aria-hidden="true"
    />
  );
}

/**
 * Card skeleton for loading states.
 */
export function CardSkeleton() {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 animate-pulse">
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 bg-gray-200 rounded-lg" />
        <div className="flex-1">
          <div className="h-4 bg-gray-200 rounded w-1/2 mb-2" />
          <div className="h-3 bg-gray-200 rounded w-1/4" />
        </div>
      </div>

      {/* Content */}
      <div className="space-y-3">
        <div className="h-3 bg-gray-200 rounded w-full" />
        <div className="h-3 bg-gray-200 rounded w-5/6" />
        <div className="h-3 bg-gray-200 rounded w-4/6" />
      </div>
    </div>
  );
}

/**
 * Results skeleton for loading prediction results.
 */
export function ResultsSkeleton() {
  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      {/* Header Skeleton */}
      <div className="bg-linear-to-r from-gray-200 to-gray-300 px-6 py-4 animate-pulse">
        <div className="h-6 bg-gray-400/30 rounded w-1/3 mb-2" />
        <div className="h-4 bg-gray-400/30 rounded w-1/2" />
      </div>

      {/* Content Skeleton */}
      <div className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Image */}
          <div className="lg:col-span-1">
            <div className="bg-gray-100 rounded-lg border border-gray-200 overflow-hidden animate-pulse">
              <div className="aspect-square bg-gray-200" />
              <div className="p-4 space-y-2">
                <div className="h-4 bg-gray-200 rounded w-3/4" />
                <div className="h-3 bg-gray-200 rounded w-1/2" />
              </div>
            </div>
          </div>

          {/* Right Column - Attributes */}
          <div className="lg:col-span-2 space-y-6">
            {/* Group 1 */}
            <div>
              <div className="h-5 bg-gray-200 rounded w-1/4 mb-3" />
              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
                <CardSkeleton />
                <CardSkeleton />
                <CardSkeleton />
              </div>
            </div>

            {/* Group 2 */}
            <div>
              <div className="h-5 bg-gray-200 rounded w-1/3 mb-3" />
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <CardSkeleton />
                <CardSkeleton />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Dots loading animation.
 */
export function LoadingDots({
  color = 'blue',
  size = 'md',
}: Pick<LoadingSpinnerProps, 'color' | 'size'>) {
  const dotSizeClasses = {
    sm: 'w-1 h-1',
    md: 'w-2 h-2',
    lg: 'w-3 h-3',
    xl: 'w-4 h-4',
  };

  return (
    <div className="flex items-center justify-center gap-1" role="status" aria-label="Loading">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className={`
            rounded-full
            ${dotSizeClasses[size]}
            ${colorClasses[color]}
            bg-current
            animate-bounce
          `}
          style={{
            animationDelay: `${i * 0.15}s`,
            animationDuration: '0.6s',
          }}
        />
      ))}
      <span className="sr-only">Loading...</span>
    </div>
  );
}

/**
 * Progress bar loading animation.
 */
export function ProgressBar({
  progress,
  color = 'blue',
  showPercentage = true,
  size = 'md',
}: {
  progress: number;
  color?: 'blue' | 'green' | 'red';
  showPercentage?: boolean;
  size?: 'sm' | 'md' | 'lg';
}) {
  // Clamp progress between 0 and 100
  const clampedProgress = Math.min(100, Math.max(0, progress));

  const heightClasses = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3',
  };

  const barColorClasses = {
    blue: 'bg-blue-600',
    green: 'bg-green-600',
    red: 'bg-red-600',
  };

  return (
    <div className="w-full" role="progressbar" aria-valuenow={clampedProgress} aria-valuemin={0} aria-valuemax={100}>
      {/* Progress Bar Container */}
      <div className={`w-full bg-gray-200 rounded-full overflow-hidden ${heightClasses[size]}`}>
        {/* Progress Bar Fill */}
        <div
          className={`h-full rounded-full transition-all duration-300 ease-out ${barColorClasses[color]}`}
          style={{ width: `${clampedProgress}%` }}
        />
      </div>

      {/* Percentage Text */}
      {showPercentage && (
        <p className="text-xs text-gray-600 mt-1 text-right">
          {Math.round(clampedProgress)}%
        </p>
      )}
    </div>
  );
}