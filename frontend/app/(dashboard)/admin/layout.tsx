/**
 * Admin Layout
 * ============
 * Admin sidebar layout for platform administrators.
 * Nested inside the dashboard route group at /admin/*.
 */

"use client";

import { AdminSidebar } from "@/components/layout/admin-sidebar";

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-gray-50 flex">
      <AdminSidebar />
      <main className="flex-1 p-6">{children}</main>
    </div>
  );
}
