/**
 * Auth Store (Zustand)
 * ====================
 * Manages authentication state: user, tenant, tokens.
 * Uses persist middleware to keep accessToken/refreshToken across page navigations.
 *
 * On page reload, the store rehydrates tokens from localStorage but NOT user/tenant.
 * The hydrateSession() method fetches the user profile from /auth/me using the
 * persisted access token, restoring the full identity.
 *
 * Also sets a cookie `sb-access-token` on login/signup so the Next.js
 * middleware (which runs server-side and can only read cookies, not
 * localStorage) can detect authenticated users and skip the login redirect.
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { User, Tenant, SignupData, InviteRegisterData } from "./types";

// ---------------------------------------------------------------------------
// Cookie helpers
// ---------------------------------------------------------------------------

const AUTH_COOKIE_NAME = "sb-access-token";

/**
 * Set a cookie that the Next.js middleware can read.
 * Expires when the browser session ends (no max-age) — this is intentional
 * because the real token expiry is managed by Supabase JWT expiry.
 */
function setAuthCookie(token: string): void {
  if (typeof document === "undefined") return; // SSR guard
  document.cookie = `${AUTH_COOKIE_NAME}=${token}; path=/; SameSite=Lax`;
}

/**
 * Remove the auth cookie on logout.
 */
function removeAuthCookie(): void {
  if (typeof document === "undefined") return;
  document.cookie = `${AUTH_COOKIE_NAME}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT`;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface AuthState {
  user: User | null;
  tenant: Tenant | null;
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  login: (email: string, password: string) => Promise<void>;
  signup: (data: SignupData) => Promise<void>;
  signupInvite: (data: InviteRegisterData) => Promise<void>;
  logout: () => Promise<void>;
  refreshSession: () => Promise<void>;
  hydrateSession: () => Promise<void>;
  setAuth: (user: User, tenant: Tenant, token: string) => void;
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

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      tenant: null,
      accessToken: null,
      refreshToken: null,
      isAuthenticated: false,
      isLoading: false,

      login: async (email: string, password: string) => {
        set({ isLoading: true });
        try {
          const api = getApiClient();
          const response = await api.login(email, password);
          setAuthCookie(response.access_token);
          set({
            user: response.user,
            tenant: response.tenant,
            accessToken: response.access_token,
            refreshToken: response.refresh_token,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      signup: async (data: SignupData) => {
        set({ isLoading: true });
        try {
          const api = getApiClient();
          const response = await api.signup(data);
          setAuthCookie(response.access_token);
          set({
            user: response.user,
            tenant: response.tenant,
            accessToken: response.access_token,
            refreshToken: response.refresh_token,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      signupInvite: async (data: InviteRegisterData) => {
        set({ isLoading: true });
        try {
          const api = getApiClient();
          const response = await api.registerViaInvite(data);
          setAuthCookie(response.access_token);
          set({
            user: response.user,
            tenant: response.tenant,
            accessToken: response.access_token,
            refreshToken: response.refresh_token,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      logout: async () => {
        try {
          const api = getApiClient();
          await api.logout();
        } catch {
          // Ignore logout errors — clear local state regardless
        } finally {
          removeAuthCookie();
          set({
            user: null,
            tenant: null,
            accessToken: null,
            refreshToken: null,
            isAuthenticated: false,
          });
        }
      },

      refreshSession: async () => {
        try {
          const api = getApiClient();
          const response = await api.refreshToken();
          setAuthCookie(response.access_token);
          set({
            accessToken: response.access_token,
            refreshToken: response.refresh_token,
          });
        } catch {
          // Refresh failed — force logout
          removeAuthCookie();
          get().logout();
        }
      },

      /**
       * Rehydrate the user/tenant profile from the backend.
       *
       * Called on app mount when the store has an accessToken but no user
       * (i.e., the page was reloaded and only tokens were persisted).
       * Uses the /auth/me endpoint which validates the JWT and returns
       * the full user + tenant profile.
       *
       * If the token is expired, attempts a refresh first. If that also
       * fails, forces a logout.
       */
      hydrateSession: async () => {
        const { accessToken, refreshToken, user } = get();

        // Nothing to hydrate if no token or already hydrated
        if (!accessToken) {
          set({ isAuthenticated: false });
          return;
        }
        if (user) {
          // Already have user data — just mark as authenticated
          set({ isAuthenticated: true });
          return;
        }

        try {
          const api = getApiClient();
          const response = await api.getMe();
          set({
            user: response.user,
            tenant: response.tenant,
            isAuthenticated: true,
          });
        } catch (error: any) {
          // If 401, try refreshing the token first
          if (error?.response?.status === 401 && refreshToken) {
            try {
              await get().refreshSession();
              // Retry /auth/me with the new token
              const api = getApiClient();
              const response = await api.getMe();
              set({
                user: response.user,
                tenant: response.tenant,
                isAuthenticated: true,
              });
              return;
            } catch {
              // Refresh also failed — logout
            }
          }
          // Token is invalid or expired and can't be refreshed
          removeAuthCookie();
          set({
            user: null,
            tenant: null,
            accessToken: null,
            refreshToken: null,
            isAuthenticated: false,
          });
        }
      },

      setAuth: (user: User, tenant: Tenant, token: string) => {
        setAuthCookie(token);
        set({
          user,
          tenant,
          accessToken: token,
          isAuthenticated: true,
        });
      },
    }),
    {
      name: "auth-storage",
      partialize: (state) => ({
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
        // Don't persist user/tenant — re-fetch on hydration
      }),
    }
  )
);
