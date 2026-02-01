'use client';

/**
 * Main Page Component
 * ===================
 * This is the main page of the application.
 * It orchestrates the upload and prediction flow.
 */

import { useState, useCallback, useEffect } from 'react';
import ImageUpload from '@/components/ImageUpload';
import PredictionResults from '@/components/PredictionResults';
import LoadingSpinner from '@/components/LoadingSpinner';
import { apiService } from '@/lib/api';
import { PredictionResult, UploadFile } from '@/lib/types';
import { AlertCircle, CheckCircle, Info } from 'lucide-react';

export default function Home() {
  // State management
  const [uploadedFiles, setUploadedFiles] = useState<UploadFile[]>([]);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  /**
   * Check API health on component mount.
   * This verifies the backend is accessible.
   */
  useEffect(() => {
    checkApiHealth();
  });

  /**
   * Check if API is accessible.
   */
  const checkApiHealth = async () => {
    try {
      await apiService.healthCheck();
      setApiStatus('online');
    } catch (error) {
      setApiStatus('offline');
      setError('Cannot connect to API server. Please make sure it is running.');
      console.error('API health check failed:', error);
    }
  };

  /**
   * Handle file selection from upload component.
   * 
   * @param files - Array of uploaded files with previews
   */
  const handleFilesSelected = useCallback((files: UploadFile[]) => {
    setUploadedFiles(files);
    setPredictions([]); // Clear previous predictions
    setError(null);
  }, []);

  /**
   * Handle prediction request.
   * This sends files to the API and displays results.
   */
  const handlePredict = async () => {
    // Validate we have files
    if (uploadedFiles.length === 0) {
      setError('Please upload at least one image');
      return;
    }

    // Reset state
    setIsLoading(true);
    setError(null);
    setPredictions([]);
    setProcessingTime(null);

    try {
      // Extract File objects from UploadFile
      const files = uploadedFiles.map(uf => uf.file);

      console.log(`Starting prediction for ${files.length} file(s)...`);

      // Make API request
      const response = await apiService.predict(files);

      console.log('Prediction successful:', response);

      // Update state with results
      setPredictions(response.predictions);
      setProcessingTime(response.processing_time_ms || null);

      // Scroll to results
      setTimeout(() => {
        document.getElementById('results-section')?.scrollIntoView({
          behavior: 'smooth',
          block: 'start',
        });
      }, 100);

    } catch (err) {
      console.error('Prediction error:', err);
      
      // Set error message
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred during prediction');
      }

      // Clear predictions on error
      setPredictions([]);

    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Handle clearing all files and results.
   */
  const handleClear = () => {
    setUploadedFiles([]);
    setPredictions([]);
    setError(null);
    setProcessingTime(null);
  };

  return (
    <main className="min-h-screen bg-linear-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Multi-Modal Classification
              </h1>
              <p className="mt-1 text-sm text-gray-600">
                Analyze advertisement images with AI-powered attribute detection
              </p>
            </div>
            
            {/* API Status Indicator */}
            <div className="flex items-center gap-2">
              <div className={`
                w-3 h-3 rounded-full
                ${apiStatus === 'online' ? 'bg-green-500' : 
                  apiStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500'}
              `} />
              <span className="text-sm text-gray-600">
                {apiStatus === 'online' ? 'API Online' : 
                 apiStatus === 'offline' ? 'API Offline' : 'Checking...'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* API Offline Warning */}
        {apiStatus === 'offline' && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 mt-0.5" />
              <div className="flex-1">
                <h3 className="text-sm font-medium text-red-800">
                  API Connection Error
                </h3>
                <p className="mt-1 text-sm text-red-700">
                  Cannot connect to the backend API. Please make sure the server is running at{' '}
                  <code className="bg-red-100 px-1 rounded">
                    {apiService.getConfig().baseUrl}
                  </code>
                </p>
                <button
                  onClick={checkApiHealth}
                  className="mt-2 text-sm text-red-800 hover:text-red-900 font-medium underline"
                >
                  Retry Connection
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Info Box */}
        <div className="mb-8 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <Info className="w-5 h-5 text-blue-600 mt-0.5" />
            <div className="flex-1">
              <h3 className="text-sm font-medium text-blue-900">
                How it works
              </h3>
              <p className="mt-1 text-sm text-blue-700">
                Upload advertisement images to analyze them for 9 different attributes including sentiment, 
                emotion, theme, target audience, and more. The AI will also extract text, keywords, 
                and promotional information from your images.
              </p>
            </div>
          </div>
        </div>

        {/* Upload Section */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Upload Images
          </h2>
          
          <ImageUpload
            onFilesSelected={handleFilesSelected}
            disabled={isLoading}
          />

          {/* Action Buttons */}
          {uploadedFiles.length > 0 && (
            <div className="mt-6 flex items-center justify-between">
              <p className="text-sm text-gray-600">
                {uploadedFiles.length} {uploadedFiles.length === 1 ? 'image' : 'images'} ready to analyze
              </p>
              
              <div className="flex gap-3">
                <button
                  onClick={handleClear}
                  disabled={isLoading}
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Clear All
                </button>
                
                <button
                  onClick={handlePredict}
                  disabled={isLoading || apiStatus === 'offline'}
                  className="px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm"
                >
                  {isLoading ? 'Analyzing...' : 'Analyze Images'}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="bg-white rounded-lg shadow-md p-12">
            <LoadingSpinner />
            <p className="text-center text-gray-600 mt-4">
              Analyzing your images... This may take a few moments.
            </p>
          </div>
        )}

        {/* Error Display */}
        {error && !isLoading && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-8">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 shrink-0" />
              <div className="flex-1">
                <h3 className="text-sm font-medium text-red-800">
                  Error
                </h3>
                <p className="mt-1 text-sm text-red-700">
                  {error}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Success Message */}
        {predictions.length > 0 && !isLoading && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-8">
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 shrink-0" />
              <div className="flex-1">
                <h3 className="text-sm font-medium text-green-800">
                  Analysis Complete
                </h3>
                <p className="mt-1 text-sm text-green-700">
                  Successfully analyzed {predictions.length} {predictions.length === 1 ? 'image' : 'images'}
                  {processingTime && ` in ${(processingTime / 1000).toFixed(2)} seconds`}.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Results Section */}
        {predictions.length > 0 && !isLoading && (
          <div id="results-section">
            <PredictionResults
              predictions={predictions}
              uploadedFiles={uploadedFiles}
            />
          </div>
        )}

        {/* Empty State */}
        {uploadedFiles.length === 0 && predictions.length === 0 && !isLoading && !error && (
          <div className="bg-white rounded-lg shadow-md p-12 text-center">
            <svg
              className="mx-auto h-12 w-12 text-gray-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
            <h3 className="mt-4 text-lg font-medium text-gray-900">
              No images uploaded
            </h3>
            <p className="mt-2 text-sm text-gray-600">
              Upload images above to get started with AI-powered analysis
            </p>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-600">
            Multi-Modal Classification System • Powered by FG_MFN
          </p>
        </div>
      </footer>
    </main>
  );
}