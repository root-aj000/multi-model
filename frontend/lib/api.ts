/**
 * API Service for Multi-Modal Classification
 * ==========================================
 * Rewritten with auth interceptors for JWT and API key authentication.
 *
 * Features:
 * - Auto-attaches JWT Bearer token from auth store
 * - Auto-refreshes expired tokens on 401 (but NOT for auth endpoints)
 * - All new endpoints: auth, history, analytics, api-keys, admin
 */

import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from "axios";
import type {
  PredictionResponse,
  ApiConfig,
  AuthResponse,
  RefreshResponse,
  SignupData,
  PredictionRecord,
  PredictionDetail,
  PaginatedResponse,
  HistoryParams,
  AnalyticsSummary,
  AttributeDistributions,
  ApiKeyCreateResponse,
  ApiKey,
  CreateKeyData,
  TestKeyResult,
  AdminTenant,
  AdminTenantDetail,
  Invitation,
  InviteVerifyResponse,
  InviteRegisterData,
  TeamMember,
} from "./types";

// ---------------------------------------------------------------------------
// Default configuration
// ---------------------------------------------------------------------------

const DEFAULT_CONFIG: ApiConfig = {
  baseUrl: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  timeout: 60000,
  maxFileSize: 10 * 1024 * 1024,
  maxFilesPerRequest: 10,
  allowedExtensions: ["jpg", "jpeg", "png", "bmp", "webp"],
};

// ---------------------------------------------------------------------------
// Auth endpoint paths — skip refresh retry for these
// ---------------------------------------------------------------------------

const AUTH_ENDPOINTS = ["/auth/login", "/auth/register", "/auth/refresh", "/auth/logout"];

function isAuthEndpoint(url: string | undefined): boolean {
  if (!url) return false;
  return AUTH_ENDPOINTS.some((ep) => url.includes(ep));
}

// ---------------------------------------------------------------------------
// API Service Class
// ---------------------------------------------------------------------------

export class ApiService {
  private client: AxiosInstance;
  private config: ApiConfig;

  constructor(config: Partial<ApiConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    this.client = axios.create({
      baseURL: this.config.baseUrl,
      timeout: this.config.timeout,
      headers: { Accept: "application/json" },
    });

    // ── Request interceptor: attach JWT ──────────────────────────────
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        // Lazy import to avoid circular dependency
        try {
          const { useAuthStore } = require("./auth");
          const token = useAuthStore.getState().accessToken;
          if (token) {
            config.headers.Authorization = `Bearer ${token}`;
          }
        } catch {
          // Auth store not available yet — skip
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // ── Response interceptor: auto-refresh on 401 ────────────────────
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;

        // Only retry once, skip auth endpoints, and only if we had a token
        if (
          error.response?.status === 401 &&
          !originalRequest._retry &&
          !isAuthEndpoint(originalRequest.url)
        ) {
          // Check if we actually have a token to refresh
          try {
            const { useAuthStore } = require("./auth");
            const currentToken = useAuthStore.getState().accessToken;
            if (!currentToken) {
              // No token to refresh — don't loop, just reject
              return Promise.reject(error);
            }

            originalRequest._retry = true;
            await useAuthStore.getState().refreshSession();
            const newToken = useAuthStore.getState().accessToken;
            originalRequest.headers.Authorization = `Bearer ${newToken}`;
            return this.client(originalRequest);
          } catch (refreshError) {
            // Refresh failed — force logout
            try {
              const { useAuthStore } = require("./auth");
              useAuthStore.getState().logout();
            } catch {
              // Ignore
            }
            if (typeof window !== "undefined") {
              window.location.href = "/login";
            }
            return Promise.reject(refreshError);
          }
        }

        return Promise.reject(error);
      }
    );
  }

  // -----------------------------------------------------------------------
  // Auth Endpoints
  // -----------------------------------------------------------------------

  async login(email: string, password: string): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>("/auth/login", {
      email,
      password,
    });
    return response.data;
  }

  async signup(data: SignupData): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>("/auth/register", data);
    return response.data;
  }

  async refreshToken(): Promise<RefreshResponse> {
    // Get refresh token from cookie or storage
    const refreshToken = this._getRefreshToken();
    const response = await this.client.post<RefreshResponse>("/auth/refresh", {
      refresh_token: refreshToken,
    });
    return response.data;
  }

  async logout(): Promise<void> {
    await this.client.post("/auth/logout");
  }

  async getMe(): Promise<AuthResponse> {
    const response = await this.client.get<AuthResponse>("/auth/me");
    return response.data;
  }

  // -----------------------------------------------------------------------
  // Prediction Endpoints
  // -----------------------------------------------------------------------

  async predict(files: File[]): Promise<PredictionResponse> {
    this.validateFiles(files);

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    const response = await this.client.post<PredictionResponse>(
      "/predict",
      formData,
      {
        headers: { "Content-Type": "multipart/form-data" },
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

    return response.data;
  }

  // -----------------------------------------------------------------------
  // History Endpoints
  // -----------------------------------------------------------------------

  async getHistory(
    params: HistoryParams = {}
  ): Promise<PaginatedResponse<PredictionRecord>> {
    const response = await this.client.get<PaginatedResponse<PredictionRecord>>(
      "/history",
      { params }
    );
    return response.data;
  }

  async getPrediction(id: string): Promise<PredictionDetail> {
    const response = await this.client.get<PredictionDetail>(`/history/${id}`);
    return response.data;
  }

  async deletePrediction(id: string): Promise<void> {
    await this.client.delete(`/history/${id}`);
  }

  // -----------------------------------------------------------------------
  // Analytics Endpoints
  // -----------------------------------------------------------------------

  async getAnalyticsSummary(): Promise<AnalyticsSummary> {
    const response = await this.client.get<AnalyticsSummary>("/analytics/summary");
    return response.data;
  }

  async getAnalyticsAttributes(): Promise<AttributeDistributions> {
    const response = await this.client.get<AttributeDistributions>("/analytics/attributes");
    return response.data;
  }

  // -----------------------------------------------------------------------
  // API Key Endpoints
  // -----------------------------------------------------------------------

  async createApiKey(data: CreateKeyData): Promise<ApiKeyCreateResponse> {
    const response = await this.client.post<ApiKeyCreateResponse>("/api-keys", data);
    return response.data;
  }

  async getApiKeys(): Promise<ApiKey[]> {
    const response = await this.client.get<{ items: ApiKey[] }>("/api-keys");
    return response.data.items;
  }

  async revokeApiKey(id: string): Promise<void> {
    await this.client.delete(`/api-keys/${id}`);
  }

  async testApiKey(id: string): Promise<TestKeyResult> {
    const response = await this.client.post<TestKeyResult>(`/api-keys/${id}/test`);
    return response.data;
  }

  // -----------------------------------------------------------------------
  // Invite Endpoints
  // -----------------------------------------------------------------------

  async createInvite(email: string, role = "member"): Promise<Invitation> {
    const response = await this.client.post<Invitation>("/invites", {
      email,
      role,
    });
    return response.data;
  }

  async listInvites(): Promise<Invitation[]> {
    const response = await this.client.get<Invitation[]>("/invites");
    return response.data;
  }

  async cancelInvite(id: string): Promise<void> {
    await this.client.delete(`/invites/${id}`);
  }

  async verifyInvite(token: string): Promise<InviteVerifyResponse> {
    const response = await this.client.get<InviteVerifyResponse>("/invites/verify", {
      params: { token },
    });
    return response.data;
  }

  async registerViaInvite(data: InviteRegisterData): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>("/auth/register-invite", data);
    return response.data;
  }

  // -----------------------------------------------------------------------
  // Team Member Endpoints
  // -----------------------------------------------------------------------

  async listMembers(): Promise<TeamMember[]> {
    const response = await this.client.get<TeamMember[]>("/invites/members");
    return response.data;
  }

  async removeMember(userId: string): Promise<void> {
    await this.client.delete(`/invites/members/${userId}`);
  }

  // -----------------------------------------------------------------------
  // Admin Endpoints
  // -----------------------------------------------------------------------

  async getAdminTenants(
    page = 1,
    pageSize = 20
  ): Promise<{ items: AdminTenant[]; total: number }> {
    const response = await this.client.get<{ items: AdminTenant[]; total: number }>(
      "/admin/tenants",
      { params: { page, page_size: pageSize } }
    );
    return response.data;
  }

  async getAdminTenant(id: string): Promise<AdminTenantDetail> {
    const response = await this.client.get<AdminTenantDetail>(`/admin/tenants/${id}`);
    return response.data;
  }

  // -----------------------------------------------------------------------
  // Health Check
  // -----------------------------------------------------------------------

  async healthCheck(): Promise<{ status: string; message: string; timestamp: string }> {
    const response = await this.client.get("/health");
    return response.data;
  }

  // -----------------------------------------------------------------------
  // Utility
  // -----------------------------------------------------------------------

  getConfig(): ApiConfig {
    return { ...this.config };
  }

  private validateFile(file: File): void {
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (!extension || !this.config.allowedExtensions.includes(extension)) {
      throw new Error(
        `Invalid file type: ${file.name}. Allowed: ${this.config.allowedExtensions.join(", ")}`
      );
    }
    if (file.size > this.config.maxFileSize) {
      throw new Error(`File too large: ${file.name}`);
    }
    if (file.size === 0) {
      throw new Error(`File is empty: ${file.name}`);
    }
  }

  private validateFiles(files: File[]): void {
    if (files.length === 0) throw new Error("No files provided");
    if (files.length > this.config.maxFilesPerRequest) {
      throw new Error(`Too many files. Maximum ${this.config.maxFilesPerRequest}`);
    }
    files.forEach((file) => this.validateFile(file));
  }

  private _getRefreshToken(): string {
    // Get from persisted auth store (Zustand + localStorage)
    try {
      const { useAuthStore } = require("./auth");
      const token = useAuthStore.getState().refreshToken;
      if (token) return token;
    } catch {
      // Auth store not available yet
    }
    return "";
  }
}

// ---------------------------------------------------------------------------
// Singleton export
// ---------------------------------------------------------------------------

export const apiService = new ApiService();
