/**
 * Dashboard Layout
 * ================
 * Sidebar + topbar shell for all authenticated pages.
 * Admin routes (/admin/*) get their own layout from admin/layout.tsx.
 */

"use client";

import { usePathname } from "next/navigation";
import { Sidebar } from "@/components/layout/sidebar";
import { Topbar } from "@/components/layout/topbar";
import { useUIStore } from "@/lib/ui-store";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const sidebarOpen = useUIStore((s) => s.sidebarOpen);

  // Admin routes have their own layout with AdminSidebar
  const isAdminRoute = pathname.startsWith("/admin");

  if (isAdminRoute) {
    return <>{children}</>;
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
      <Sidebar />
      <div
        className={`transition-all duration-200 ${
          sidebarOpen ? "ml-64" : "ml-0"
        }`}
      >
        <Topbar />
        <main className="p-6">{children}</main>
      </div>
    </div>
  );
}
