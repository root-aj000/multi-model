/**
 * Admin Dashboard Page
 * ====================
 * Platform-wide statistics and system health.
 */

"use client";

import { useEffect, useState } from "react";
import { apiService } from "@/lib/api";
import type { AdminTenant } from "@/lib/types";
import { Building2, Users, Image, Activity } from "lucide-react";

export default function AdminDashboardPage() {
  const [tenants, setTenants] = useState<AdminTenant[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const result = await apiService.getAdminTenants(1, 100);
        setTenants(result.items);
      } catch (error) {
        console.error("Failed to load admin data:", error);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-amber-400" />
      </div>
    );
  }

  const totalUsers = tenants.reduce((sum, t) => sum + t.user_count, 0);
  const totalPredictions = tenants.reduce((sum, t) => sum + t.prediction_count, 0);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Admin Dashboard</h1>
        <p className="text-gray-500 mt-1">Platform-wide statistics and management</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-primary-400/10 rounded-lg">
              <Building2 className="w-5 h-5 text-primary-400" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Tenants</p>
              <p className="text-2xl font-bold text-gray-900">{tenants.length}</p>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-green-100 rounded-lg">
              <Users className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Total Users</p>
              <p className="text-2xl font-bold text-gray-900">{totalUsers}</p>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-purple-100 rounded-lg">
              <Image className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Total Predictions</p>
              <p className="text-2xl font-bold text-gray-900">{totalPredictions}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Tenant List */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="font-semibold text-gray-900">All Tenants</h2>
        </div>
        {tenants.length === 0 ? (
          <div className="px-6 py-12 text-center text-gray-500">No tenants found</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Slug</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Plan</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Users</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Predictions</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Created</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {tenants.map((tenant) => (
                  <tr key={tenant.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4">
                      <a href={`/admin/tenants/${tenant.id}`} className="text-sm font-medium text-primary-400 hover:underline">
                        {tenant.name}
                      </a>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-600">{tenant.slug}</td>
                    <td className="px-6 py-4">
                      <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full capitalize">
                        {tenant.plan}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-600">{tenant.user_count}</td>
                    <td className="px-6 py-4 text-sm text-gray-600">{tenant.prediction_count}</td>
                    <td className="px-6 py-4 text-sm text-gray-500">
                      {new Date(tenant.created_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
