/**
 * API Service for Multi-Modal Classification
 * ==========================================
 * This service handles all communication with the backend API.
 * It provides type-safe methods for making predictions and checking health.
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import { PredictionResponse, ErrorResponse, ApiConfig } from './types';

/**
 * Default API configuration.
 * Adjust these values based on your environment.
 */
const DEFAULT_CONFIG: ApiConfig = {
  baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  timeout: 60000, // 60 seconds
  maxFileSize: 10 * 1024 * 1024, // 10 MB
  maxFilesPerRequest: 10,
  allowedExtensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp'],
};

/**
 * API Service Class
 * 
 * This class encapsulates all API interactions with proper error handling,
 * timeout management, and type safety.
 * 
 * Usage:
 *   const api = new ApiService();
 *   const results = await api.predict(files);
 */
export class ApiService {
  private client: AxiosInstance;
  private config: ApiConfig;

  /**
   * Initialize the API service.
   * 
   * @param config - Optional custom configuration
   */
  constructor(config: Partial<ApiConfig> = {}) {
    // Merge custom config with defaults
    this.config = { ...DEFAULT_CONFIG, ...config };

    // Create axios instance with base configuration
    this.client = axios.create({
      baseURL: this.config.baseUrl,
      timeout: this.config.timeout,
      headers: {
        'Accept': 'application/json',
      },
    });

    // Add request interceptor for logging (development only)
    if (process.env.NODE_ENV === 'development') {
      this.client.interceptors.request.use(
        (config) => {
          console.log('🚀 API Request:', config.method?.toUpperCase(), config.url);
          return config;
        },
        (error) => {
          console.error('❌ Request Error:', error);
          return Promise.reject(error);
        }
      );
    }

    // Add response interceptor for logging (development only)
    if (process.env.NODE_ENV === 'development') {
      this.client.interceptors.response.use(
        (response) => {
          console.log('✅ API Response:', response.status, response.config.url);
          return response;
        },
        (error) => {
          console.error('❌ Response Error:', error.response?.status, error.config?.url);
          return Promise.reject(error);
        }
      );
    }
  }

  /**
   * Validate file before upload.
   * 
   * Checks:
   * - File extension is allowed
   * - File size is within limit
   * 
   * @param file - File to validate
   * @throws Error if validation fails
   */
  private validateFile(file: File): void {
    // Check file extension
    const extension = file.name.split('.').pop()?.toLowerCase();
    
    if (!extension || !this.config.allowedExtensions.includes(extension)) {
      throw new Error(
        `Invalid file type: ${file.name}. Allowed types: ${this.config.allowedExtensions.join(', ')}`
      );
    }

    // Check file size
    if (file.size > this.config.maxFileSize) {
      const maxSizeMB = this.config.maxFileSize / (1024 * 1024);
      const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
      throw new Error(
        `File too large: ${file.name} (${fileSizeMB}MB). Maximum size: ${maxSizeMB}MB`
      );
    }

    // Check if file is actually a file
    if (file.size === 0) {
      throw new Error(`File is empty: ${file.name}`);
    }
  }

  /**
   * Validate array of files before upload.
   * 
   * @param files - Array of files to validate
   * @throws Error if validation fails
   */
  private validateFiles(files: File[]): void {
    // Check number of files
    if (files.length === 0) {
      throw new Error('No files provided');
    }

    if (files.length > this.config.maxFilesPerRequest) {
      throw new Error(
        `Too many files. Maximum ${this.config.maxFilesPerRequest} files allowed, got ${files.length}`
      );
    }

    // Validate each file
    files.forEach((file) => this.validateFile(file));
  }

  /**
   * Make predictions for uploaded images.
   * 
   * This is the main method for getting predictions from the API.
   * It handles file validation, upload, and error handling.
   * 
   * @param files - Array of image files to analyze
   * @returns Promise with prediction results
   * @throws Error if prediction fails
   * 
   * @example
   * ```typescript
   * const api = new ApiService();
   * const files = [imageFile1, imageFile2];
   * 
   * try {
   *   const results = await api.predict(files);
   *   console.log(results.predictions);
   * } catch (error) {
   *   console.error('Prediction failed:', error.message);
   * }
   * ```
   */
  async predict(files: File[]): Promise<PredictionResponse> {
    try {
      // Step 1: Validate files
      console.log(`📤 Uploading ${files.length} file(s)...`);
      this.validateFiles(files);
      console.log('✓ File validation passed');

      // Step 2: Create FormData
      // FormData is used for multipart/form-data uploads
      const formData = new FormData();
      
      files.forEach((file) => {
        formData.append('files', file);
      });

      console.log('✓ FormData created');

      // Step 3: Make API request
      console.log('🔄 Sending request to API...');
      const startTime = Date.now();

      const response = await this.client.post<PredictionResponse>(
        '/predict',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          // Upload progress callback (optional)
          onUploadProgress: (progressEvent) => {
            if (progressEvent.total) {
              const percentCompleted = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              );
              console.log(`Upload progress: ${percentCompleted}%`);
            }
          },
        }
      );

      const endTime = Date.now();
      const totalTime = endTime - startTime;

      console.log('✅ Prediction successful');
      console.log(`⏱️  Total time: ${totalTime}ms`);
      console.log(`📊 Results: ${response.data.total_images} image(s)`);

      // Step 4: Return results
      return response.data;

    } catch (error) {
      // Handle different types of errors
      if (axios.isAxiosError(error)) {
        const axiosError = error as AxiosError<ErrorResponse>;

        // Server returned an error response
        if (axiosError.response) {
          const errorDetail = axiosError.response.data?.detail || 'Unknown error';
          console.error('❌ Server error:', errorDetail);
          throw new Error(`Server error: ${errorDetail}`);
        }

        // Request was made but no response received
        if (axiosError.request) {
          console.error('❌ Network error: No response from server');
          throw new Error(
            'Network error: Could not connect to server. Please check if the API is running.'
          );
        }
      }

      // Request setup error or validation error
      if (error instanceof Error) {
        console.error('❌ Error:', error.message);
        throw error;
      }

      // Unknown error
      console.error('❌ Unknown error:', error);
      throw new Error('An unexpected error occurred');
    }
  }

  /**
   * Check API health status.
   * 
   * This method pings the /health endpoint to verify the API is running.
   * Useful for displaying connection status in the UI.
   * 
   * @returns Promise with health status
   * @throws Error if health check fails
   * 
   * @example
   * ```typescript
   * const api = new ApiService();
   * 
   * try {
   *   const health = await api.healthCheck();
   *   console.log('API Status:', health.status);
   * } catch (error) {
   *   console.error('API is down');
   * }
   * ```
   */
  async healthCheck(): Promise<{ status: string; message: string; timestamp: string }> {
    try {
      console.log('🏥 Checking API health...');

      const response = await this.client.get('/health');

      console.log('✅ API is healthy:', response.data.status);

      return response.data;

    } catch (error) {
      console.error('❌ Health check failed');

      if (axios.isAxiosError(error)) {
        throw new Error('API is not responding. Please check if the server is running.');
      }

      throw new Error('Health check failed');
    }
  }

  /**
   * Get API configuration.
   * 
   * Returns the current API configuration for display or validation purposes.
   * 
   * @returns Current API configuration
   */
  getConfig(): ApiConfig {
    return { ...this.config };
  }
}

/**
 * Create a singleton instance of the API service.
 * This is the recommended way to use the API service in your application.
 * 
 * Usage:
 *   import { apiService } from '@/lib/api';
 *   const results = await apiService.predict(files);
 */
export const apiService = new ApiService();

/**
 * Export default instance for convenience.
 */
export default apiService;