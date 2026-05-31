/**
 * Type Definitions for Multi-Modal Classification API
 * ===================================================
 * These types define the structure of data exchanged with the API.
 * They ensure type safety throughout the application.
 */

// ---------------------------------------------------------------------------
// Prediction Types (existing)
// ---------------------------------------------------------------------------

/**
 * Single prediction result for one image.
 * Contains all 9 attributes plus extracted features.
 */
export interface PredictionResult {
  // Basic info
  id?: string;  // UUID — only present when auth is enabled
  filename: string;
  predicted_label: string;

  // OCR result
  ocr_text: string;

  // 9 Attributes (all optional as they depend on model output)
  theme?: string;
  sentiment?: string;
  emotion?: string;
  dominant_colour?: string;
  attention_score?: string;
  trust_safety?: string;
  target_audience?: string;
  predicted_ctr?: string;
  likelihood_shares?: string;

  // Extracted text features
  keywords?: string;
  monetary_mention?: string;
  call_to_action?: string;
  object_detected?: string;
}

/**
 * API response structure for predictions.
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
  timestamp?: string;
}

/**
 * Upload file with preview.
 */
export interface UploadFile {
  file: File;
  preview: string;
  id: string;
}

/**
 * API configuration.
 */
export interface ApiConfig {
  baseUrl: string;
  timeout: number;
  maxFileSize: number;
  maxFilesPerRequest: number;
  allowedExtensions: string[];
}

// ---------------------------------------------------------------------------
// Auth Types
// ---------------------------------------------------------------------------

/**
 * User data returned from auth endpoints.
 */
export interface User {
  id: string;
  email: string;
  display_name: string | null;
  role: "owner" | "admin" | "member" | "platform_admin";
}

/**
 * Tenant data returned from auth endpoints.
 */
export interface Tenant {
  id: string;
  name: string;
  slug: string;
  plan: "free" | "pro" | "enterprise";
}

/**
 * Registration request data.
 */
export interface SignupData {
  email: string;
  password: string;
  display_name: string;
  tenant_name: string;
  tenant_slug: string;
}

/**
 * Auth response from login/register endpoints.
 */
export interface AuthResponse {
  access_token: string;
  refresh_token: string | null;
  token_type: string;
  expires_in: number;
  user: User;
  tenant: Tenant;
}

/**
 * Token refresh response.
 */
export interface RefreshResponse {
  access_token: string;
  refresh_token: string | null;
  token_type: string;
  expires_in: number;
}

// ---------------------------------------------------------------------------
// History Types
// ---------------------------------------------------------------------------

/**
 * Prediction record in history list.
 */
export interface PredictionRecord {
  id: string;
  filename: string | null;
  predicted_label: string | null;
  theme?: string;
  sentiment?: string;
  processing_ms: number | null;
  created_at: string;
}

/**
 * Full prediction detail.
 */
export interface PredictionDetail {
  id: string;
  filename: string | null;
  ocr_text: string | null;
  result: Record<string, unknown>;
  processing_ms: number | null;
  user_id: string;
  created_at: string;
}

/**
 * Paginated response wrapper.
 */
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

/**
 * History query parameters.
 */
export interface HistoryParams {
  page?: number;
  page_size?: number;
  attribute?: string;
  value?: string;
  search?: string;
}

// ---------------------------------------------------------------------------
// Analytics Types
// ---------------------------------------------------------------------------

/**
 * Analytics summary for the dashboard.
 */
export interface AnalyticsSummary {
  total_predictions: number;
  predictions_this_week: number;
  predictions_this_month: number;
  avg_processing_ms: number | null;
  most_common_theme: string | null;
  most_common_sentiment: string | null;
  quota_used: number;
  quota_limit: number;
}

/**
 * Attribute distribution counts.
 */
export interface AttributeDistributions {
  theme: Record<string, number>;
  sentiment: Record<string, number>;
  emotion: Record<string, number>;
  dominant_colour: Record<string, number>;
  attention_score: Record<string, number>;
  trust_safety: Record<string, number>;
  target_audience: Record<string, number>;
  predicted_ctr: Record<string, number>;
  likelihood_shares: Record<string, number>;
}

// ---------------------------------------------------------------------------
// API Key Types
// ---------------------------------------------------------------------------

/**
 * API key data returned after creation (includes full key ONE TIME ONLY).
 */
export interface ApiKeyCreateResponse {
  id: string;
  name: string;
  key: string;  // Only shown once!
  key_prefix: string;
  permissions: string[];
  expires_at: string | null;
  created_at: string;
}

/**
 * API key item in the list.
 */
export interface ApiKey {
  id: string;
  name: string;
  key_prefix: string;
  permissions: string[];
  last_used_at: string | null;
  expires_at: string | null;
  revoked_at: string | null;
  created_at: string;
}

/**
 * Request to create a new API key.
 */
export interface CreateKeyData {
  name: string;
  permissions: string[];
  expires_in_days: number | null;
}

/**
 * Result of testing an API key.
 */
export interface TestKeyResult {
  valid: boolean;
  status: number;
  response_time_ms: number;
  tested_at: string;
  response_body: Record<string, unknown> | null;
}

// ---------------------------------------------------------------------------
// Admin Types
// ---------------------------------------------------------------------------

/**
 * Tenant item in admin list.
 */
export interface AdminTenant {
  id: string;
  name: string;
  slug: string;
  plan: string;
  user_count: number;
  prediction_count: number;
  created_at: string;
}

/**
 * Full tenant detail for admin view.
 */
export interface AdminTenantDetail {
  id: string;
  name: string;
  slug: string;
  plan: string;
  settings: Record<string, unknown>;
  monthly_limit: number;
  user_count: number;
  prediction_count: number;
  created_at: string;
  updated_at: string | null;
}

// ---------------------------------------------------------------------------
// Invite / Team Types
// ---------------------------------------------------------------------------

/**
 * Invitation record.
 */
export interface Invitation {
  id: string;
  email: string;
  role: string;
  status: "pending" | "accepted" | "expired" | "cancelled";
  expires_at: string;
  created_at: string;
  invite_link?: string | null;
}

/**
 * Response from verifying an invite token.
 */
export interface InviteVerifyResponse {
  valid: boolean;
  email?: string;
  tenant_name?: string;
  role?: string;
  error?: string;
}

/**
 * Request to register via invitation.
 */
export interface InviteRegisterData {
  email: string;
  password: string;
  display_name: string;
  token: string;
}

/**
 * Team member data.
 */
export interface TeamMember {
  id: string;
  email: string;
  display_name: string | null;
  role: "owner" | "admin" | "member";
}
