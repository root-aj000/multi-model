/**
 * UI Store (Zustand)
 * ==================
 * Manages UI state: sidebar visibility, active tenant.
 */

import { create } from "zustand";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface UIState {
  sidebarOpen: boolean;
  activeTenantId: string | null;

  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  setActiveTenant: (id: string) => void;
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const useUIStore = create<UIState>()((set) => ({
  sidebarOpen: true,
  activeTenantId: null,

  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setSidebarOpen: (open: boolean) => set({ sidebarOpen: open }),
  setActiveTenant: (id: string) => set({ activeTenantId: id }),
}));
