/**
 * API Key Detail Page
 * ===================
 * Shows key details, usage stats, and integration code snippets.
 */

"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { apiService } from "@/lib/api";
import type { ApiKey, TestKeyResult } from "@/lib/types";
import { ArrowLeft, CheckCircle2, XCircle, Clock, Loader2, Shield } from "lucide-react";

export default function ApiKeyDetailPage() {
  const params = useParams();
  const id = params.id as string;

  const [key, setKey] = useState<ApiKey | null>(null);
  const [testResult, setTestResult] = useState<TestKeyResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [testing, setTesting] = useState(false);

  useEffect(() => {
    async function loadKey() {
      try {
        const keys = await apiService.getApiKeys();
        const found = keys.find((k) => k.id === id);
        setKey(found || null);
      } catch (error) {
        console.error("Failed to load key:", error);
      } finally {
        setLoading(false);
      }
    }
    loadKey();
  }, [id]);

  const handleTest = async () => {
    setTesting(true);
    try {
      const result = await apiService.testApiKey(id);
      setTestResult(result);
    } catch (error) {
      console.error("Failed to test key:", error);
    } finally {
      setTesting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-400" />
      </div>
    );
  }

  if (!key) {
    return (
      <div className="text-center py-12 text-gray-500">API key not found</div>
    );
  }

  const isActive = !key.revoked_at && (!key.expires_at || new Date(key.expires_at) > new Date());

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <a href="/api-keys" className="p-2 hover:bg-gray-100 rounded-lg">
          <ArrowLeft className="w-5 h-5 text-gray-600" />
        </a>
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{key.name}</h1>
          <p className="text-sm text-gray-500 mt-1">
            <code className="bg-gray-100 px-1 rounded">{key.key_prefix}••••••</code>
            {" · "}
            {isActive ? (
              <span className="text-green-600">Active</span>
            ) : key.revoked_at ? (
              <span className="text-red-600">Revoked</span>
            ) : (
              <span className="text-amber-600">Expired</span>
            )}
          </p>
        </div>
      </div>

      {/* Key Info */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="divide-y divide-gray-100">
          <div className="px-6 py-4 flex justify-between">
            <span className="text-sm text-gray-500">Key Prefix</span>
            <code className="text-sm font-mono text-gray-900">{key.key_prefix}••••••</code>
          </div>
          <div className="px-6 py-4 flex justify-between">
            <span className="text-sm text-gray-500">Permissions</span>
            <span className="text-sm text-gray-900">{key.permissions.join(", ")}</span>
          </div>
          <div className="px-6 py-4 flex justify-between">
            <span className="text-sm text-gray-500">Created</span>
            <span className="text-sm text-gray-900">{new Date(key.created_at).toLocaleDateString()}</span>
          </div>
          {key.expires_at && (
            <div className="px-6 py-4 flex justify-between">
              <span className="text-sm text-gray-500">Expires</span>
              <span className="text-sm text-gray-900">{new Date(key.expires_at).toLocaleDateString()}</span>
            </div>
          )}
          {key.last_used_at && (
            <div className="px-6 py-4 flex justify-between">
              <span className="text-sm text-gray-500">Last Used</span>
              <span className="text-sm text-gray-900">{new Date(key.last_used_at).toLocaleDateString()}</span>
            </div>
          )}
        </div>
      </div>

      {/* Test Key */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-semibold text-gray-900">Test Key</h2>
          <button
            onClick={handleTest}
            disabled={testing || !isActive}
            className="flex items-center gap-2 px-4 py-2 bg-primary-400 text-white rounded-lg hover:bg-primary-500 disabled:opacity-50 transition-colors"
          >
            {testing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Shield className="w-4 h-4" />}
            Test Key
          </button>
        </div>
        {testResult && (
          <div className={`p-4 rounded-lg ${testResult.valid ? "bg-green-50 border border-green-200" : "bg-red-50 border border-red-200"}`}>
            <div className="flex items-center gap-2">
              {testResult.valid ? (
                <CheckCircle2 className="w-5 h-5 text-green-600" />
              ) : (
                <XCircle className="w-5 h-5 text-red-600" />
              )}
              <span className={`font-medium ${testResult.valid ? "text-green-800" : "text-red-800"}`}>
                {testResult.valid ? "Key is valid" : "Key is not valid"}
              </span>
            </div>
            <p className="text-sm mt-2 text-gray-600">
              Status: {testResult.status} · Response time: {testResult.response_time_ms}ms
            </p>
          </div>
        )}
      </div>

      {/* Integration Code */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="font-semibold text-gray-900 mb-4">Integration Code</h2>
        <div className="space-y-4">
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">Python</p>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-xs overflow-x-auto">
{`import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}"

response = requests.post(
    f"{BASE_URL}/predict",
    headers={"X-API-Key": API_KEY},
    files={"files": open("image.jpg", "rb")}
)
result = response.json()
print(f"Theme: {result['predictions'][0]['theme']}")`}
            </pre>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">cURL</p>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-xs overflow-x-auto">
{`curl -X POST ${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/predict \\
  -H "X-API-Key: YOUR_API_KEY" \\
  -F "files=@image.jpg"`}
            </pre>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">JavaScript</p>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-xs overflow-x-auto">
{`const API_KEY = "YOUR_API_KEY";
const BASE_URL = "${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}";

const formData = new FormData();
formData.append("files", fileInput.files[0]);

const response = await fetch(\`\${BASE_URL}/predict\`, {
  method: "POST",
  headers: { "X-API-Key": API_KEY },
  body: formData,
});
const data = await response.json();`}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
