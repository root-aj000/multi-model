/**
 * Admin Tenants List Page
 * =======================
 */

"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { apiService } from "@/lib/api";
import type { AdminTenant } from "@/lib/types";
import { Building2, Loader2 } from "lucide-react";

export default function AdminTenantsPage() {
  const [tenants, setTenants] = useState<AdminTenant[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const result = await apiService.getAdminTenants(1, 100);
        setTenants(result.items);
      } catch (error) {
        console.error("Failed to load tenants:", error);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-6 h-6 animate-spin text-amber-400" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Tenants</h1>
        <p className="text-gray-500 mt-1">{tenants.length} workspaces</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {tenants.map((tenant) => (
          <Link
            key={tenant.id}
            href={`/admin/tenants/${tenant.id}`}
            className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:border-primary-400/30 hover:shadow-md transition-all"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-primary-400/10 rounded-lg flex items-center justify-center">
                <Building2 className="w-5 h-5 text-primary-400" />
              </div>
              <div>
                <p className="font-medium text-gray-900">{tenant.name}</p>
                <p className="text-xs text-gray-500">{tenant.slug}</p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <p className="text-gray-500">Plan</p>
                <p className="font-medium capitalize">{tenant.plan}</p>
              </div>
              <div>
                <p className="text-gray-500">Users</p>
                <p className="font-medium">{tenant.user_count}</p>
              </div>
              <div>
                <p className="text-gray-500">Predictions</p>
                <p className="font-medium">{tenant.prediction_count}</p>
              </div>
              <div>
                <p className="text-gray-500">Created</p>
                <p className="font-medium">{new Date(tenant.created_at).toLocaleDateString()}</p>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
