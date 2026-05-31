/**
 * Dashboard Sidebar Navigation — Premium Glass Design
 * ====================================================
 * Glass effect, #ff6b35 accent, interactive nav items with borders + animations.
 */

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useUIStore } from "@/lib/ui-store";
import { useAuthStore } from "@/lib/auth";
import {
  BarChart3,
  History,
  Image,
  Key,
  LayoutDashboard,
  Settings,
  ChevronLeft,
  Sparkles,
} from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/predict", label: "Predict", icon: Image },
  { href: "/history", label: "History", icon: History },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/api-keys", label: "API Keys", icon: Key },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const sidebarOpen = useUIStore((s) => s.sidebarOpen);
  const toggleSidebar = useUIStore((s) => s.toggleSidebar);
  const tenant = useAuthStore((s) => s.tenant);

  if (!sidebarOpen) return null;

  return (
    <aside className="fixed inset-y-0 left-0 z-30 w-64 glass-sidebar flex flex-col">
      {/* Header */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-white/10 dark:border-white/5">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 bg-gradient-to-br from-primary-400 to-primary-600 rounded-xl flex items-center justify-center shadow-glow border border-primary-400/50">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <p className="text-sm font-semibold text-gray-900 dark:text-gray-100 truncate max-w-[150px]">
              {tenant?.name || "Workspace"}
            </p>
            <p className="text-[11px] text-gray-400 dark:text-gray-500 capitalize">{tenant?.plan || "free"} plan</p>
          </div>
        </div>
        <button
          onClick={toggleSidebar}
          className="p-1.5 rounded-lg border border-transparent hover:border-gray-200 dark:hover:border-gray-700 hover:bg-black/5 dark:hover:bg-white/5 transition-all duration-200 active:scale-90"
          aria-label="Close sidebar"
        >
          <ChevronLeft className="w-5 h-5 text-gray-400 dark:text-gray-500" />
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);
          const Icon = item.icon;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 border ${
                isActive
                  ? "bg-primary-400/10 text-primary-500 dark:bg-primary-400/15 dark:text-primary-400 shadow-sm border-primary-400/20 dark:border-primary-400/15"
                  : "text-gray-500 hover:bg-black/5 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-white/5 dark:hover:text-gray-200 border-transparent hover:border-gray-200 dark:hover:border-gray-700"
              }`}
            >
              <Icon className="w-5 h-5" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-white/10 dark:border-white/5">
        <p className="text-[11px] text-gray-400 dark:text-gray-600">
          Multi-Model Classification v2.0
        </p>
      </div>
    </aside>
  );
}
