/**
 * Invite Registration Page
 * ========================
 * Page for users who received an email invitation.
 * Validates the invite token, then shows a registration form
 * with email pre-filled (read-only).
 */

"use client";

import { useState, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { useAuthStore } from "@/lib/auth";
import { apiService } from "@/lib/api";
import type { InviteVerifyResponse } from "@/lib/types";
import { AlertCircle, Loader2, UserPlus, Mail, Building2 } from "lucide-react";

export default function InvitePage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const token = searchParams.get("token") || "";

  const signupInvite = useAuthStore((s) => s.signupInvite);
  const isLoading = useAuthStore((s) => s.isLoading);

  const [verifyState, setVerifyState] = useState<{
    loading: boolean;
    verified: boolean;
    data: InviteVerifyResponse | null;
  }>({ loading: true, verified: false, data: null });

  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  // ── Verify token on mount ──────────────────────────────────────────────
  useEffect(() => {
    if (!token) {
      setVerifyState({ loading: false, verified: false, data: { valid: false, error: "No invitation token provided." } });
      return;
    }

    const verify = async () => {
      try {
        const result = await apiService.verifyInvite(token);
        setVerifyState({ loading: false, verified: result.valid, data: result });
        if (result.valid && result.email) {
          setEmail(result.email);
        }
      } catch {
        setVerifyState({
          loading: false,
          verified: false,
          data: { valid: false, error: "Failed to verify invitation. Please try again." },
        });
      }
    };

    verify();
  }, [token]);

  // ── Handle registration submit ────────────────────────────────────────
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (password.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }

    try {
      await signupInvite({
        email,
        password,
        display_name: displayName,
        token,
      });
      router.push("/");
    } catch (err: any) {
      const msg =
        err?.response?.data?.detail ||
        (err instanceof Error ? err.message : "Registration failed");
      setError(msg);
    }
  };

  // ── Loading state ────────────────────────────────────────────────────
  if (verifyState.loading) {
    return (
      <div className="glass rounded-2xl shadow-glass p-8 border border-white/20 dark:border-white/5">
        <div className="flex flex-col items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-primary-400 mb-4" />
          <p className="text-gray-500 dark:text-gray-400">Verifying your invitation...</p>
        </div>
      </div>
    );
  }

  // ── Invalid/expired token ────────────────────────────────────────────
  if (!verifyState.verified && verifyState.data) {
    return (
      <div className="glass rounded-2xl shadow-glass p-8 border border-white/20 dark:border-white/5">
        <div className="text-center mb-6">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-100 dark:bg-red-900/20 flex items-center justify-center">
            <AlertCircle className="w-8 h-8 text-red-500" />
          </div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Invalid Invitation</h1>
          <p className="text-gray-500 dark:text-gray-400 mt-2">
            {verifyState.data.error || "This invitation is no longer valid."}
          </p>
        </div>
        <div className="space-y-3">
          <Link
            href="/login"
            className="block w-full text-center px-4 py-2.5 bg-primary-400 text-white rounded-xl hover:bg-primary-500 transition-colors font-medium"
          >
            Sign in instead
          </Link>
          <Link
            href="/signup"
            className="block w-full text-center px-4 py-2.5 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors font-medium"
          >
            Create a new workspace
          </Link>
        </div>
      </div>
    );
  }

  // ── Valid token — show registration form ──────────────────────────────
  return (
    <div className="glass rounded-2xl shadow-glass p-8 border border-white/20 dark:border-white/5">
      <div className="text-center mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Accept your invitation</h1>
        <p className="text-gray-500 dark:text-gray-400 mt-2">
          Create your account to join the workspace
        </p>
      </div>

      {/* Workspace info */}
      {verifyState.data?.tenant_name && (
        <div className="mb-6 p-3 bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 rounded-xl flex items-center gap-3">
          <Building2 className="w-5 h-5 text-primary-500 shrink-0" />
          <div>
            <p className="text-sm font-medium text-primary-700 dark:text-primary-300">
              Joining: {verifyState.data.tenant_name}
            </p>
            <p className="text-xs text-primary-500 dark:text-primary-400">
              Role: {verifyState.data.role || "member"}
            </p>
          </div>
        </div>
      )}

      {error && (
        <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-center gap-2 text-sm text-red-700 dark:text-red-400 animate-scale-in">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Email (read-only) */}
        <div>
          <label htmlFor="email" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Email
          </label>
          <div className="relative">
            <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              id="email"
              type="email"
              value={email}
              readOnly
              className="w-full pl-10 pr-3 py-2.5 glass-input rounded-xl bg-gray-50 dark:bg-gray-800 text-gray-500 dark:text-gray-400 cursor-not-allowed focus:outline-none"
            />
          </div>
        </div>

        {/* Display name */}
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

        {/* Password */}
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
          {isLoading ? "Creating account..." : "Accept invitation & create account"}
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
