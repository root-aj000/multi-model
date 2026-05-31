/**
 * Admin Tenant Detail Page
 * ========================
 */

"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { apiService } from "@/lib/api";
import type { AdminTenantDetail } from "@/lib/types";
import { ArrowLeft, Building2, Loader2, Users, Image, Settings } from "lucide-react";

export default function AdminTenantDetailPage() {
  const params = useParams();
  const id = params.id as string;

  const [tenant, setTenant] = useState<AdminTenantDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadTenant() {
      try {
        const data = await apiService.getAdminTenant(id);
        setTenant(data);
      } catch (error) {
        console.error("Failed to load tenant:", error);
      } finally {
        setLoading(false);
      }
    }
    loadTenant();
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-6 h-6 animate-spin text-amber-400" />
      </div>
    );
  }

  if (!tenant) {
    return (
      <div className="text-center py-12 text-gray-500">
        <p>Tenant not found</p>
        <Link href="/admin/tenants" className="text-primary-400 hover:underline mt-2 inline-block">
          Back to Tenants
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Link href="/admin/tenants" className="p-2 hover:bg-gray-100 rounded-lg">
          <ArrowLeft className="w-5 h-5 text-gray-600" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{tenant.name}</h1>
          <p className="text-sm text-gray-500">{tenant.slug}</p>
        </div>
        <span className="px-3 py-1 bg-gray-100 text-gray-600 text-sm font-medium rounded-full capitalize">
          {tenant.plan}
        </span>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-primary-400/10 rounded-lg">
              <Users className="w-5 h-5 text-primary-400" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Users</p>
              <p className="text-2xl font-bold text-gray-900">{tenant.user_count}</p>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-green-100 rounded-lg">
              <Image className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Predictions</p>
              <p className="text-2xl font-bold text-gray-900">{tenant.prediction_count}</p>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-purple-100 rounded-lg">
              <Settings className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Monthly Limit</p>
              <p className="text-2xl font-bold text-gray-900">{tenant.monthly_limit}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Details */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="font-semibold text-gray-900">Tenant Details</h2>
        </div>
        <div className="divide-y divide-gray-100">
          <div className="px-6 py-4 flex justify-between">
            <span className="text-sm text-gray-500">ID</span>
            <code className="text-sm font-mono text-gray-900">{tenant.id}</code>
          </div>
          <div className="px-6 py-4 flex justify-between">
            <span className="text-sm text-gray-500">Plan</span>
            <span className="text-sm text-gray-900 capitalize">{tenant.plan}</span>
          </div>
          <div className="px-6 py-4 flex justify-between">
            <span className="text-sm text-gray-500">Monthly Limit</span>
            <span className="text-sm text-gray-900">{tenant.monthly_limit}</span>
          </div>
          <div className="px-6 py-4 flex justify-between">
            <span className="text-sm text-gray-500">Created</span>
            <span className="text-sm text-gray-900">{new Date(tenant.created_at).toLocaleString()}</span>
          </div>
          {tenant.updated_at && (
            <div className="px-6 py-4 flex justify-between">
              <span className="text-sm text-gray-500">Updated</span>
              <span className="text-sm text-gray-900">{new Date(tenant.updated_at).toLocaleString()}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
