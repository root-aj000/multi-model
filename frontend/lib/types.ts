/**
 * Type Definitions for Multi-Modal Classification API
 * ===================================================
 * These types define the structure of data exchanged with the API.
 * They ensure type safety throughout the application.
 */

/**
 * Single prediction result for one image.
 * Contains all 9 attributes plus extracted features.
 * Note: No confidence scores as per API design.
 */
export interface PredictionResult {
  // Basic info
  filename: string;
  predicted_label: string;
  
  // OCR result
  ocr_text: string;
  
  // 9 Attributes (all optional as they depend on model output)
  theme?: string;              // Topic or theme of the content
  sentiment?: string;          // Positive/Negative/Neutral sentiment
  emotion?: string;            // Specific emotion (happy, sad, angry, etc.)
  dominant_colour?: string;    // Main color in the image
  attention_score?: string;    // How attention-grabbing the content is
  trust_safety?: string;       // Safety and trustworthiness level
  target_audience?: string;    // Intended audience type
  predicted_ctr?: string;      // Click-through rate prediction
  likelihood_shares?: string;  // Likelihood of being shared
  
  // Extracted text features
  keywords?: string;           // Important keywords from text
  monetary_mention?: string;   // Price/discount information
  call_to_action?: string;     // CTA phrases detected
  object_detected?: string;    // Product categories detected
}

/**
 * API response structure for predictions.
 * Contains array of results plus metadata.
 */
export interface PredictionResponse {
  predictions: PredictionResult[];
  total_images: number;
  processing_time_ms?: number;
}

/**
 * Error response structure from API.
 */
export interface ErrorResponse {
  detail: string;
  error_code?: string;
  timestamp: string;
}

/**
 * Upload file with preview.
 * Used internally to track uploaded files before sending to API.
 */
export interface UploadFile {
  file: File;
  preview: string;  // Data URL for preview
  id: string;       // Unique identifier
}

/**
 * API configuration.
 */
export interface ApiConfig {
  baseUrl: string;
  timeout: number;
  maxFileSize: number;      // In bytes
  maxFilesPerRequest: number;
  allowedExtensions: string[];
}