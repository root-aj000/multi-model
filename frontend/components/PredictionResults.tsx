'use client';

/**
 * Prediction Results Component
 * ============================
 * Displays prediction results for all uploaded images.
 * Shows all 9 attributes, OCR text, and extracted features.
 * 
 * Features:
 * - Tabbed view for multiple images
 * - Grouped attributes by category
 * - OCR text display
 * - Extracted features display
 * - Responsive design
 */

import { useState } from 'react';
import { PredictionResult, UploadFile } from '@/lib/types';
import AttributeCard from '../components/AttributeCard';
import Image from 'next/image';
import {
  FileText,
  Tag,
  Eye,
  Users,
  TrendingUp,
  Share2,
  Shield,
  Palette,
  Heart,
  MessageSquare,
  Search,
  DollarSign,
  MousePointer,
  Package,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';

interface PredictionResultsProps {
  predictions: PredictionResult[];
  uploadedFiles: UploadFile[];
}

export default function PredictionResults({
  predictions,
  uploadedFiles,
}: PredictionResultsProps) {
  // State for current image index (for navigation)
  const [currentIndex, setCurrentIndex] = useState(0);

  // Get current prediction
  const currentPrediction = predictions[currentIndex];

  // Find matching uploaded file for preview
  const currentFile = uploadedFiles.find(
    (file) => file.file.name === currentPrediction?.filename
  ) || uploadedFiles[currentIndex];

  /**
   * Navigate to previous image.
   */
  const handlePrevious = () => {
    setCurrentIndex((prev) => (prev > 0 ? prev - 1 : predictions.length - 1));
  };

  /**
   * Navigate to next image.
   */
  const handleNext = () => {
    setCurrentIndex((prev) => (prev < predictions.length - 1 ? prev + 1 : 0));
  };

  // If no predictions, show nothing
  if (!predictions || predictions.length === 0) {
    return null;
  }

  // Group attributes for display
  const attributeGroups = [
    {
      title: 'Content Analysis',
      description: 'What the image is about',
      attributes: [
        {
          key: 'theme',
          label: 'Theme',
          icon: Tag,
          color: 'blue',
        },
        {
          key: 'sentiment',
          label: 'Sentiment',
          icon: Heart,
          color: 'pink',
        },
        {
          key: 'emotion',
          label: 'Emotion',
          icon: MessageSquare,
          color: 'purple',
        },
      ],
    },
    {
      title: 'Visual Analysis',
      description: 'Visual characteristics',
      attributes: [
        {
          key: 'dominant_colour',
          label: 'Dominant Color',
          icon: Palette,
          color: 'orange',
        },
        {
          key: 'attention_score',
          label: 'Attention Score',
          icon: Eye,
          color: 'yellow',
        },
      ],
    },
    {
      title: 'Audience & Performance',
      description: 'Target audience and predicted performance',
      attributes: [
        {
          key: 'target_audience',
          label: 'Target Audience',
          icon: Users,
          color: 'green',
        },
        {
          key: 'predicted_ctr',
          label: 'Predicted CTR',
          icon: TrendingUp,
          color: 'cyan',
        },
        {
          key: 'likelihood_shares',
          label: 'Share Likelihood',
          icon: Share2,
          color: 'indigo',
        },
      ],
    },
    {
      title: 'Trust & Safety',
      description: 'Safety and trustworthiness assessment',
      attributes: [
        {
          key: 'trust_safety',
          label: 'Trust & Safety',
          icon: Shield,
          color: 'emerald',
        },
      ],
    },
  ];

  // Extracted features for display
  const extractedFeatures = [
    {
      key: 'keywords',
      label: 'Keywords',
      icon: Search,
      color: 'gray',
    },
    {
      key: 'monetary_mention',
      label: 'Price/Discount',
      icon: DollarSign,
      color: 'green',
    },
    {
      key: 'call_to_action',
      label: 'Call to Action',
      icon: MousePointer,
      color: 'blue',
    },
    {
      key: 'object_detected',
      label: 'Objects Detected',
      icon: Package,
      color: 'purple',
    },
  ];

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      {/* Header */}
      <div className="bg-linear-to-r from-blue-600 to-purple-600 px-6 py-4">
        <h2 className="text-xl font-semibold text-white">
          Prediction Results
        </h2>
        <p className="text-blue-100 text-sm mt-1">
          AI-powered analysis of your advertisement images
        </p>
      </div>

      {/* Image Navigation (only if multiple images) */}
      {predictions.length > 1 && (
        <div className="border-b border-gray-200 bg-gray-50 px-6 py-3">
          <div className="flex items-center justify-between">
            {/* Previous Button */}
            <button
              onClick={handlePrevious}
              className="flex items-center gap-1 px-3 py-1.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
              Previous
            </button>

            {/* Image Indicator */}
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600">
                Image {currentIndex + 1} of {predictions.length}
              </span>
              
              {/* Dot Indicators */}
              <div className="flex gap-1">
                {predictions.map((_, index) => (
                  <button
                    key={index}
                    onClick={() => setCurrentIndex(index)}
                    className={`
                      w-2 h-2 rounded-full transition-colors
                      ${index === currentIndex
                        ? 'bg-blue-600'
                        : 'bg-gray-300 hover:bg-gray-400'}
                    `}
                    title={`Go to image ${index + 1}`}
                  />
                ))}
              </div>
            </div>

            {/* Next Button */}
            <button
              onClick={handleNext}
              className="flex items-center gap-1 px-3 py-1.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-100 transition-colors"
            >
              Next
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Left Column - Image Preview */}
          <div className="lg:col-span-1">
            {/* Image Preview Card */}
            <div className="bg-gray-50 rounded-lg border border-gray-200 overflow-hidden">
              {/* Image */}
              <div className="aspect-square bg-gray-100">
                {currentFile && (
                  <Image
                    src={currentFile.preview}
                    alt={currentPrediction.filename}
                    className="w-full h-full object-contain"
                  />
                )}
              </div>

              {/* Image Info */}
              <div className="p-4 border-t border-gray-200">
                <h3 className="font-medium text-gray-900 truncate" title={currentPrediction.filename}>
                  {currentPrediction.filename}
                </h3>
                
                {/* Primary Prediction */}
                <div className="mt-3 flex items-center gap-2">
                  <span className="text-sm text-gray-600">Primary Label:</span>
                  <span className="px-2 py-0.5 text-sm font-medium text-blue-700 bg-blue-100 rounded-full">
                    {currentPrediction.predicted_label || 'Unknown'}
                  </span>
                </div>
              </div>
            </div>

            {/* OCR Text Card */}
            <div className="mt-4 bg-gray-50 rounded-lg border border-gray-200 p-4">
              <div className="flex items-center gap-2 mb-3">
                <FileText className="w-5 h-5 text-gray-600" />
                <h3 className="font-medium text-gray-900">Extracted Text (OCR)</h3>
              </div>
              
              <div className="bg-white rounded border border-gray-200 p-3 min-h-20">
                {currentPrediction.ocr_text ? (
                  <p className="text-sm text-gray-700 whitespace-pre-wrap">
                    {currentPrediction.ocr_text}
                  </p>
                ) : (
                  <p className="text-sm text-gray-400 italic">
                    No text detected in this image
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Right Column - Attributes */}
          <div className="lg:col-span-2 space-y-6">
            
            {/* Attribute Groups */}
            {attributeGroups.map((group) => {
              // Filter out attributes that don't have values
              const availableAttributes = group.attributes.filter(
                (attr) => currentPrediction[attr.key as keyof PredictionResult]
              );

              // Skip group if no attributes have values
              if (availableAttributes.length === 0) {
                return null;
              }

              return (
                <div key={group.title}>
                  {/* Group Header */}
                  <div className="mb-3">
                    <h3 className="text-lg font-semibold text-gray-900">
                      {group.title}
                    </h3>
                    <p className="text-sm text-gray-600">
                      {group.description}
                    </p>
                  </div>

                  {/* Attribute Cards */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
                    {availableAttributes.map((attr) => (
                      <AttributeCard
                        key={attr.key}
                        label={attr.label}
                        value={currentPrediction[attr.key as keyof PredictionResult] as string}
                        icon={attr.icon}
                        color={attr.color}
                      />
                    ))}
                  </div>
                </div>
              );
            })}

            {/* Extracted Features */}
            <div>
              <div className="mb-3">
                <h3 className="text-lg font-semibold text-gray-900">
                  Extracted Features
                </h3>
                <p className="text-sm text-gray-600">
                  Information extracted from the image text
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {extractedFeatures.map((feature) => {
                  const value = currentPrediction[feature.key as keyof PredictionResult] as string;
                  
                  // Skip if no value or "None"
                  if (!value || value === 'None' || value === '') {
                    return (
                      <AttributeCard
                        key={feature.key}
                        label={feature.label}
                        value="Not detected"
                        icon={feature.icon}
                        color="gray"
                        muted
                      />
                    );
                  }

                  return (
                    <AttributeCard
                      key={feature.key}
                      label={feature.label}
                      value={value}
                      icon={feature.icon}
                      color={feature.color}
                    />
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer - Quick Stats */}
      {predictions.length > 1 && (
        <div className="border-t border-gray-200 bg-gray-50 px-6 py-4">
          <h4 className="text-sm font-medium text-gray-900 mb-3">
            Summary Across All Images
          </h4>
          
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {/* Total Images */}
            <div className="bg-white rounded-lg border border-gray-200 p-3 text-center">
              <p className="text-2xl font-bold text-blue-600">{predictions.length}</p>
              <p className="text-xs text-gray-600">Images Analyzed</p>
            </div>

            {/* Most Common Sentiment */}
            <div className="bg-white rounded-lg border border-gray-200 p-3 text-center">
              <p className="text-lg font-bold text-pink-600 truncate">
                {getMostCommon(predictions, 'sentiment')}
              </p>
              <p className="text-xs text-gray-600">Common Sentiment</p>
            </div>

            {/* Most Common Theme */}
            <div className="bg-white rounded-lg border border-gray-200 p-3 text-center">
              <p className="text-lg font-bold text-purple-600 truncate">
                {getMostCommon(predictions, 'theme')}
              </p>
              <p className="text-xs text-gray-600">Common Theme</p>
            </div>

            {/* Most Common Audience */}
            <div className="bg-white rounded-lg border border-gray-200 p-3 text-center">
              <p className="text-lg font-bold text-green-600 truncate">
                {getMostCommon(predictions, 'target_audience')}
              </p>
              <p className="text-xs text-gray-600">Target Audience</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Helper function to get the most common value for an attribute.
 */
function getMostCommon(predictions: PredictionResult[], key: keyof PredictionResult): string {
  // Count occurrences
  const counts: Record<string, number> = {};
  
  predictions.forEach((pred) => {
    const value = pred[key] as string;
    if (value && value !== 'None' && value !== '') {
      counts[value] = (counts[value] || 0) + 1;
    }
  });

  // Find most common
  let maxCount = 0;
  let mostCommon = 'N/A';

  Object.entries(counts).forEach(([value, count]) => {
    if (count > maxCount) {
      maxCount = count;
      mostCommon = value;
    }
  });

  return mostCommon;
}