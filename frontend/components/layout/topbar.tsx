/**
 * Dashboard Top Bar — Premium Glass Design
 * ==========================================
 * Glass effect, interactive buttons with borders + animations.
 */

"use client";

import { useAuthStore } from "@/lib/auth";
import { useUIStore } from "@/lib/ui-store";
import { useTheme } from "next-themes";
import { Menu, LogOut, User, Sun, Moon } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export function Topbar() {
  const user = useAuthStore((s) => s.user);
  const tenant = useAuthStore((s) => s.tenant);
  const logout = useAuthStore((s) => s.logout);
  const toggleSidebar = useUIStore((s) => s.toggleSidebar);
  const sidebarOpen = useUIStore((s) => s.sidebarOpen);
  const { theme, setTheme } = useTheme();

  const handleLogout = async () => {
    await logout();
    window.location.href = "/login";
  };

  return (
    <header className="h-16 glass-topbar flex items-center justify-between px-6 sticky top-0 z-20">
      <div className="flex items-center gap-4">
        {!sidebarOpen && (
          <button
            onClick={toggleSidebar}
            className="p-2 rounded-xl border border-transparent hover:border-gray-200 dark:hover:border-gray-700 hover:bg-black/5 dark:hover:bg-white/5 transition-all duration-200 active:scale-90"
            aria-label="Open sidebar"
          >
            <Menu className="w-5 h-5 text-gray-500 dark:text-gray-400" />
          </button>
        )}
      </div>

      <div className="flex items-center gap-3">
        {/* Tenant badge */}
        {tenant && (
          <Badge variant={tenant.plan === "pro" ? "pro" : tenant.plan === "enterprise" ? "enterprise" : "free"}>
            {tenant.plan}
          </Badge>
        )}

        {/* Dark mode toggle */}
        <button
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          className="p-2 rounded-xl border border-transparent hover:border-gray-200 dark:hover:border-gray-700 hover:bg-black/5 dark:hover:bg-white/5 transition-all duration-200 active:scale-90"
          aria-label="Toggle dark mode"
        >
          {theme === "dark" ? (
            <Sun className="w-5 h-5 text-amber-400" />
          ) : (
            <Moon className="w-5 h-5 text-gray-400" />
          )}
        </button>

        {/* User info */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-gradient-to-br from-primary-400 to-primary-600 rounded-full flex items-center justify-center shadow-glow border border-primary-400/50">
            <User className="w-4 h-4 text-white" />
          </div>
          <div className="hidden sm:block">
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
              {user?.display_name || user?.email || "User"}
            </p>
            <p className="text-xs text-gray-400 dark:text-gray-500 capitalize">{user?.role || "member"}</p>
          </div>
        </div>

        {/* Logout */}
        <button
          onClick={handleLogout}
          className="p-2 rounded-xl border border-transparent hover:border-red-200 dark:hover:border-red-800 hover:bg-red-50 dark:hover:bg-red-900/20 transition-all duration-200 active:scale-90 group"
          aria-label="Logout"
          title="Sign out"
        >
          <LogOut className="w-5 h-5 text-gray-400 dark:text-gray-500 group-hover:text-red-500 transition-colors" />
        </button>
      </div>
    </header>
  );
}
