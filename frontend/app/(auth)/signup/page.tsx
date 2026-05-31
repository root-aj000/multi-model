/**
 * Signup Page — Premium Design
 * =============================
 * Registration form with tenant creation, #ff6b35 primary button.
 */

"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuthStore } from "@/lib/auth";
import { AlertCircle, Loader2, UserPlus } from "lucide-react";

export default function SignupPage() {
  const router = useRouter();
  const signup = useAuthStore((s) => s.signup);
  const isLoading = useAuthStore((s) => s.isLoading);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [tenantName, setTenantName] = useState("");
  const [tenantSlug, setTenantSlug] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleSlugFromName = (name: string) => {
    const slug = name
      .toLowerCase()
      .replace(/[^a-z0-9-]/g, "-")
      .replace(/-+/g, "-")
      .replace(/^-|-$/g, "");
    setTenantSlug(slug);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (password.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }

    try {
      await signup({
        email,
        password,
        display_name: displayName,
        tenant_name: tenantName,
        tenant_slug: tenantSlug,
      });
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed");
    }
  };

  return (
    <div className="glass rounded-2xl shadow-glass p-8 border border-white/20 dark:border-white/5">
      <div className="text-center mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Create your account</h1>
        <p className="text-gray-500 dark:text-gray-400 mt-2">Get started with a new workspace</p>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-center gap-2 text-sm text-red-700 dark:text-red-400 animate-scale-in">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Personal info */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label htmlFor="displayName" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Your Name
            </label>
            <input
              id="displayName"
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              required
              className="w-full px-3 py-2.5 glass-input rounded-xl text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-400"
              placeholder="John Smith"
            />
          </div>
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Email
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full px-3 py-2.5 glass-input rounded-xl text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-400"
              placeholder="you@company.com"
            />
          </div>
        </div>

        <div>
          <label htmlFor="password" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Password
          </label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={8}
            className="w-full px-3 py-2.5 glass-input rounded-xl text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-400"
            placeholder="Min 8 characters"
          />
        </div>

        {/* Workspace info */}
        <div className="border-t border-white/10 dark:border-white/5 pt-4 mt-4">
          <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">Workspace</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="tenantName" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Workspace Name
              </label>
              <input
                id="tenantName"
                type="text"
                value={tenantName}
                onChange={(e) => {
                  setTenantName(e.target.value);
                  if (!tenantSlug) handleSlugFromName(e.target.value);
                }}
                required
                className="w-full px-3 py-2.5 glass-input rounded-xl text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-400"
                placeholder="Acme Corp"
              />
            </div>
            <div>
              <label htmlFor="tenantSlug" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                URL Slug
              </label>
              <input
                id="tenantSlug"
                type="text"
                value={tenantSlug}
                onChange={(e) => setTenantSlug(e.target.value.toLowerCase())}
                required
                pattern="[a-z0-9][a-z0-9\-]{1,98}[a-z0-9]"
                className="w-full px-3 py-2.5 glass-input rounded-xl text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-400"
                placeholder="acme-corp"
              />
            </div>
          </div>
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="btn-primary w-full justify-center"
        >
          {isLoading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <UserPlus className="w-4 h-4" />
          )}
          {isLoading ? "Creating account..." : "Create account"}
        </button>
      </form>

      <p className="mt-6 text-center text-sm text-gray-500 dark:text-gray-400">
        Already have an account?{" "}
        <Link href="/login" className="text-primary-400 hover:text-primary-500 font-medium transition-colors">
          Sign in
        </Link>
      </p>
    </div>
  );
}
