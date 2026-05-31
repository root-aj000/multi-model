/**
 * API Keys Store (Zustand)
 * ========================
 * Manages API key state: list, create, revoke, test.
 */

import { create } from "zustand";
import type { ApiKey, CreateKeyData, ApiKeyCreateResponse, TestKeyResult } from "./types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ApiKeysState {
  keys: ApiKey[];
  isLoading: boolean;
  error: string | null;

  fetchKeys: () => Promise<void>;
  createKey: (data: CreateKeyData) => Promise<ApiKeyCreateResponse>;
  revokeKey: (id: string) => Promise<void>;
  testKey: (id: string) => Promise<TestKeyResult>;
}

// ---------------------------------------------------------------------------
// API client (lazy import to avoid circular deps)
// ---------------------------------------------------------------------------

function getApiClient() {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { apiService } = require("./api");
  return apiService;
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const useApiKeysStore = create<ApiKeysState>()((set, get) => ({
  keys: [],
  isLoading: false,
  error: null,

  fetchKeys: async () => {
    set({ isLoading: true, error: null });
    try {
      const api = getApiClient();
      const keys = await api.getApiKeys();
      set({ keys, isLoading: false });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Failed to load API keys",
        isLoading: false,
      });
    }
  },

  createKey: async (data: CreateKeyData) => {
    set({ isLoading: true, error: null });
    try {
      const api = getApiClient();
      const result = await api.createApiKey(data);
      // Refresh the list after creating
      await get().fetchKeys();
      set({ isLoading: false });
      return result;
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Failed to create API key",
        isLoading: false,
      });
      throw error;
    }
  },

  revokeKey: async (id: string) => {
    set({ isLoading: true, error: null });
    try {
      const api = getApiClient();
      await api.revokeApiKey(id);
      // Refresh the list after revoking
      await get().fetchKeys();
      set({ isLoading: false });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Failed to revoke API key",
        isLoading: false,
      });
      throw error;
    }
  },

  testKey: async (id: string) => {
    try {
      const api = getApiClient();
      return await api.testApiKey(id);
    } catch (error) {
      throw error;
    }
  },
}));
