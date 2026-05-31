/**
 * API Keys Page
 * =============
 * List, create, and manage API keys for programmatic access.
 */

"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useApiKeysStore } from "@/lib/api-keys-store";
import type { CreateKeyData } from "@/lib/types";
import {
  Key,
  Plus,
  AlertCircle,
  CheckCircle2,
  Copy,
  Loader2,
  Eye,
  Shield,
  Clock,
  XCircle,
} from "lucide-react";

export default function ApiKeysPage() {
  const { keys, isLoading, fetchKeys, createKey, revokeKey } = useApiKeysStore();
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [createdKey, setCreatedKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Create form state
  const [name, setName] = useState("");
  const [permissions, setPermissions] = useState<string[]>(["predict"]);
  const [expiresIn, setExpiresIn] = useState<string>("90");
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchKeys();
  }, [fetchKeys]);

  const handleCreate = async () => {
    setIsCreating(true);
    setError(null);
    try {
      const result = await createKey({
        name,
        permissions,
        expires_in_days: expiresIn === "never" ? null : parseInt(expiresIn),
      });
      setCreatedKey(result.key);
      setName("");
      setPermissions(["predict"]);
      setExpiresIn("90");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create key");
    } finally {
      setIsCreating(false);
    }
  };

  const handleCopy = () => {
    if (createdKey) {
      navigator.clipboard.writeText(createdKey);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleRevoke = async (id: string) => {
    if (!confirm("Revoke this API key? This cannot be undone.")) return;
    try {
      await revokeKey(id);
    } catch (err) {
      console.error("Failed to revoke key:", err);
    }
  };

  const closeDialog = () => {
    setShowCreateDialog(false);
    setCreatedKey(null);
    setCopied(false);
    setError(null);
  };

  const PERMISSIONS = [
    { id: "predict", label: "Predictions", desc: "POST /predict" },
    { id: "history", label: "History", desc: "GET /history" },
    { id: "analytics", label: "Analytics", desc: "GET /analytics" },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">API Keys</h1>
          <p className="text-gray-500 mt-1">Manage programmatic access to the API</p>
        </div>
        <button
          onClick={() => setShowCreateDialog(true)}
          className="flex items-center gap-2 px-4 py-2 bg-primary-400 text-white rounded-lg hover:bg-primary-500 transition-colors"
        >
          <Plus className="w-4 h-4" /> New API Key
        </button>
      </div>

      {/* Key List */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        {isLoading ? (
          <div className="flex items-center justify-center h-32">
            <Loader2 className="w-6 h-6 animate-spin text-primary-400" />
          </div>
        ) : keys.length === 0 ? (
          <div className="px-6 py-12 text-center text-gray-500">
            <Key className="w-12 h-12 mx-auto text-gray-300 mb-3" />
            <p>No API keys yet. Create one to get started.</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {keys.map((key) => (
              <div
                key={key.id}
                className="flex items-center justify-between px-6 py-4 hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
                    <Key className="w-5 h-5 text-gray-400" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-medium text-gray-900">{key.name}</p>
                      {key.revoked_at ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-red-100 text-red-700 text-xs rounded-full">
                          <XCircle className="w-3 h-3" /> Revoked
                        </span>
                      ) : key.expires_at && new Date(key.expires_at) < new Date() ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-amber-100 text-amber-700 text-xs rounded-full">
                          <Clock className="w-3 h-3" /> Expired
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded-full">
                          <CheckCircle2 className="w-3 h-3" /> Active
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-gray-500 mt-0.5">
                      <code className="bg-gray-100 px-1 rounded">{key.key_prefix}••••••</code>
                      {" · "}
                      {key.permissions.join(", ")}
                      {key.last_used_at && ` · Last used ${new Date(key.last_used_at).toLocaleDateString()}`}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Link
                    href={`/api-keys/${key.id}`}
                    className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                    title="View details"
                  >
                    <Eye className="w-4 h-4 text-gray-400" />
                  </Link>
                  {!key.revoked_at && (
                    <button
                      onClick={() => handleRevoke(key.id)}
                      className="p-2 hover:bg-red-50 rounded-lg transition-colors"
                      title="Revoke key"
                    >
                      <Shield className="w-4 h-4 text-gray-400 hover:text-red-500" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Create Key Dialog */}
      {showCreateDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-md mx-4 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900">
                {createdKey ? "API Key Created" : "Create New API Key"}
              </h2>
              <button onClick={closeDialog} className="p-1 hover:bg-gray-100 rounded">
                <XCircle className="w-5 h-5 text-gray-400" />
              </button>
            </div>

            {createdKey ? (
              /* One-time key reveal */
              <div className="space-y-4">
                <div className="flex items-start gap-3 p-4 bg-amber-50 border border-amber-200 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-amber-600 mt-0.5 shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-amber-800">Save this key now!</p>
                    <p className="text-xs text-amber-700 mt-1">
                      This is the only time the full key will be shown. After closing, you can only
                      see the prefix.
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <code className="flex-1 p-3 bg-gray-100 rounded text-sm font-mono break-all">
                    {createdKey}
                  </code>
                  <button
                    onClick={handleCopy}
                    className="p-2 hover:bg-gray-100 rounded-lg shrink-0"
                  >
                    {copied ? (
                      <CheckCircle2 className="w-4 h-4 text-green-600" />
                    ) : (
                      <Copy className="w-4 h-4 text-gray-400" />
                    )}
                  </button>
                </div>
                <button
                  onClick={closeDialog}
                  className="w-full px-4 py-2 bg-primary-400 text-white rounded-lg hover:bg-primary-500 transition-colors"
                >
                  I've saved the key — Close
                </button>
              </div>
            ) : (
              /* Create form */
              <div className="space-y-4">
                {error && (
                  <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
                    {error}
                  </div>
                )}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Key Name</label>
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder='e.g. "Production Server"'
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-400"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Permissions</label>
                  {PERMISSIONS.map((perm) => (
                    <label key={perm.id} className="flex items-center gap-2 py-1">
                      <input
                        type="checkbox"
                        checked={permissions.includes(perm.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setPermissions([...permissions, perm.id]);
                          } else {
                            setPermissions(permissions.filter((p) => p !== perm.id));
                          }
                        }}
                        className="rounded border-gray-300"
                      />
                      <span className="text-sm">{perm.label}</span>
                      <span className="text-xs text-gray-400">({perm.desc})</span>
                    </label>
                  ))}
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Expiration</label>
                  <select
                    value={expiresIn}
                    onChange={(e) => setExpiresIn(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-400"
                  >
                    <option value="30">30 days</option>
                    <option value="90">90 days</option>
                    <option value="365">1 year</option>
                    <option value="never">Never</option>
                  </select>
                </div>
                <div className="flex gap-2 justify-end pt-2">
                  <button
                    onClick={closeDialog}
                    className="px-4 py-2 text-gray-600 hover:text-gray-800"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleCreate}
                    disabled={!name || permissions.length === 0 || isCreating}
                    className="flex items-center gap-2 px-4 py-2 bg-primary-400 text-white rounded-lg hover:bg-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {isCreating && <Loader2 className="w-4 h-4 animate-spin" />}
                    Create Key
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
