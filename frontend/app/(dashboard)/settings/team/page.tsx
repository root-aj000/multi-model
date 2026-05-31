/**
 * Team Management Page
 * =====================
 * View and manage team members (admin/owner only).
 * Features:
 * - List all team members with role badges
 * - Invite new members by email
 * - Cancel pending invitations
 * - Remove members (owner/admin only, cannot remove owner or self)
 */

"use client";

import { useState, useEffect, useCallback } from "react";
import { useAuthStore } from "@/lib/auth";
import { apiService } from "@/lib/api";
import type { TeamMember, Invitation } from "@/lib/types";
import {
  Users,
  UserPlus,
  Trash2,
  Shield,
  Mail,
  Clock,
  X,
  AlertCircle,
  Loader2,
  Crown,
  CheckCircle,
  XCircle,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Role badge component
// ---------------------------------------------------------------------------

function RoleBadge({ role }: { role: string }) {
  const config: Record<string, { label: string; className: string; icon: React.ReactNode }> = {
    owner: {
      label: "Owner",
      className: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400",
      icon: <Crown className="w-3 h-3" />,
    },
    admin: {
      label: "Admin",
      className: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
      icon: <Shield className="w-3 h-3" />,
    },
    member: {
      label: "Member",
      className: "bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300",
      icon: <Users className="w-3 h-3" />,
    },
  };

  const c = config[role] || config.member;

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${c.className}`}>
      {c.icon}
      {c.label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Invite status badge
// ---------------------------------------------------------------------------

function InviteStatusBadge({ status }: { status: string }) {
  const config: Record<string, { label: string; className: string; icon: React.ReactNode }> = {
    pending: {
      label: "Pending",
      className: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400",
      icon: <Clock className="w-3 h-3" />,
    },
    accepted: {
      label: "Accepted",
      className: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
      icon: <CheckCircle className="w-3 h-3" />,
    },
    expired: {
      label: "Expired",
      className: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
      icon: <XCircle className="w-3 h-3" />,
    },
    cancelled: {
      label: "Cancelled",
      className: "bg-gray-100 text-gray-500 dark:bg-gray-700 dark:text-gray-400",
      icon: <XCircle className="w-3 h-3" />,
    },
  };

  const c = config[status] || config.pending;

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${c.className}`}>
      {c.icon}
      {c.label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Main page component
// ---------------------------------------------------------------------------

export default function TeamPage() {
  const user = useAuthStore((s) => s.user);
  const isAdmin = user?.role === "admin" || user?.role === "owner";

  const [members, setMembers] = useState<TeamMember[]>([]);
  const [invites, setInvites] = useState<Invitation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Invite modal state
  const [showInviteModal, setShowInviteModal] = useState(false);
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState("member");
  const [inviteLoading, setInviteLoading] = useState(false);
  const [inviteError, setInviteError] = useState<string | null>(null);
  const [inviteSuccess, setInviteSuccess] = useState(false);
  const [inviteLinkResult, setInviteLinkResult] = useState<string | null>(null);

  // Delete confirmation state
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; name: string; type: "member" | "invite" } | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);

  // ── Fetch data ────────────────────────────────────────────────────────
  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [membersData, invitesData] = await Promise.all([
        apiService.listMembers(),
        apiService.listInvites(),
      ]);
      setMembers(membersData);
      setInvites(invitesData);
    } catch (err: any) {
      const msg = err?.response?.data?.detail || "Failed to load team data";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isAdmin) {
      fetchData();
    } else {
      setLoading(false);
    }
  }, [isAdmin, fetchData]);

  // ── Invite member ────────────────────────────────────────────────────
  const handleInvite = async (e: React.FormEvent) => {
    e.preventDefault();
    setInviteError(null);
    setInviteSuccess(false);
    setInviteLoading(true);
    setInviteLinkResult(null);

    try {
      const result = await apiService.createInvite(inviteEmail, inviteRole);
      if (result.invite_link) {
        // Email was NOT sent — show the link for manual sharing
        setInviteLinkResult(result.invite_link);
      } else {
        // Email was sent successfully
        setInviteSuccess(true);
        setInviteEmail("");
        setInviteRole("member");
        await fetchData();
        setTimeout(() => {
          setShowInviteModal(false);
          setInviteSuccess(false);
        }, 1500);
      }
    } catch (err: any) {
      const msg = err?.response?.data?.detail || "Failed to send invitation";
      setInviteError(msg);
    } finally {
      setInviteLoading(false);
    }
  };

  // ── Remove member ────────────────────────────────────────────────────
  const handleDeleteConfirm = async () => {
    if (!deleteTarget) return;
    setDeleteLoading(true);

    try {
      if (deleteTarget.type === "member") {
        await apiService.removeMember(deleteTarget.id);
      } else {
        await apiService.cancelInvite(deleteTarget.id);
      }
      await fetchData();
    } catch (err: any) {
      const msg = err?.response?.data?.detail || "Failed to remove";
      setError(msg);
    } finally {
      setDeleteLoading(false);
      setDeleteTarget(null);
    }
  };

  // ── Not admin ─────────────────────────────────────────────────────────
  if (!isAdmin) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Team</h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">Manage workspace members</p>
        </div>
        <div className="p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg text-sm text-amber-700 dark:text-amber-400">
          <Shield className="w-4 h-4 inline mr-2" />
          Only admins and owners can manage team members.
        </div>
      </div>
    );
  }

  // ── Loading ───────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Team</h1>
            <p className="text-gray-500 dark:text-gray-400 mt-1">Manage workspace members</p>
          </div>
        </div>
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-primary-400" />
        </div>
      </div>
    );
  }

  // ── Pending invites (only show pending) ──────────────────────────────
  const pendingInvites = invites.filter((i) => i.status === "pending");
  const pastInvites = invites.filter((i) => i.status !== "pending");

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Team</h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            {members.length} member{members.length !== 1 ? "s" : ""}
            {pendingInvites.length > 0 && ` · ${pendingInvites.length} pending invite${pendingInvites.length !== 1 ? "s" : ""}`}
          </p>
        </div>
        <button
          onClick={() => {
            setShowInviteModal(true);
            setInviteError(null);
            setInviteSuccess(false);
          }}
          className="flex items-center gap-2 px-4 py-2 bg-primary-400 text-white rounded-lg hover:bg-primary-500 transition-colors"
        >
          <UserPlus className="w-4 h-4" /> Invite Member
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-center gap-2 text-sm text-red-700 dark:text-red-400">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
          <button
            onClick={() => setError(null)}
            className="ml-auto text-red-500 hover:text-red-700 dark:hover:text-red-300"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* ── Team Members Table ────────────────────────────────────────── */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
            <Users className="w-5 h-5 text-gray-500" />
            Members
          </h2>
        </div>

        {members.length === 0 ? (
          <div className="p-12 text-center text-gray-500 dark:text-gray-400">
            <Users className="w-12 h-12 mx-auto text-gray-300 dark:text-gray-600 mb-3" />
            <p>No team members found.</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {members.map((member) => (
              <div
                key={member.id}
                className="flex items-center justify-between px-6 py-4 hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors"
              >
                <div className="flex items-center gap-4">
                  {/* Avatar */}
                  <div className="w-10 h-10 rounded-full bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center text-primary-600 dark:text-primary-400 font-semibold text-sm">
                    {(member.display_name || member.email).charAt(0).toUpperCase()}
                  </div>
                  {/* Info */}
                  <div>
                    <p className="font-medium text-gray-900 dark:text-gray-100">
                      {member.display_name || member.email.split("@")[0]}
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">{member.email}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <RoleBadge role={member.role} />
                  {/* Delete button — not for owner or self */}
                  {member.role !== "owner" && member.id !== user?.id && (
                    <button
                      onClick={() =>
                        setDeleteTarget({
                          id: member.id,
                          name: member.display_name || member.email,
                          type: "member",
                        })
                      }
                      className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
                      title="Remove member"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Pending Invitations ───────────────────────────────────────── */}
      {pendingInvites.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
              <Mail className="w-5 h-5 text-yellow-500" />
              Pending Invitations
            </h2>
          </div>
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {pendingInvites.map((invite) => (
              <div
                key={invite.id}
                className="flex items-center justify-between px-6 py-4 hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors"
              >
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-full bg-yellow-100 dark:bg-yellow-900/30 flex items-center justify-center text-yellow-600 dark:text-yellow-400">
                    <Mail className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-900 dark:text-gray-100">{invite.email}</p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Invited as {invite.role} · Expires {new Date(invite.expires_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <InviteStatusBadge status={invite.status} />
                  <button
                    onClick={() =>
                      setDeleteTarget({
                        id: invite.id,
                        name: invite.email,
                        type: "invite",
                      })
                    }
                    className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
                    title="Cancel invitation"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Past Invitations (collapsed) ──────────────────────────────── */}
      {pastInvites.length > 0 && (
        <details className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
          <summary className="px-6 py-4 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors text-sm font-medium text-gray-500 dark:text-gray-400">
            Past Invitations ({pastInvites.length})
          </summary>
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {pastInvites.map((invite) => (
              <div
                key={invite.id}
                className="flex items-center justify-between px-6 py-3 opacity-60"
              >
                <div className="flex items-center gap-3">
                  <span className="text-sm text-gray-700 dark:text-gray-300">{invite.email}</span>
                </div>
                <InviteStatusBadge status={invite.status} />
              </div>
            ))}
          </div>
        </details>
      )}

      {/* ── Invite Modal ──────────────────────────────────────────────── */}
      {showInviteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700 w-full max-w-md mx-4 p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Invite Team Member
              </h2>
              <button
                onClick={() => setShowInviteModal(false)}
                className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {inviteError && (
              <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-center gap-2 text-sm text-red-700 dark:text-red-400">
                <AlertCircle className="w-4 h-4 shrink-0" />
                {inviteError}
              </div>
            )}

            {inviteSuccess && (
              <div className="mb-4 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg flex items-center gap-2 text-sm text-green-700 dark:text-green-400">
                <CheckCircle className="w-4 h-4 shrink-0" />
                Invitation sent successfully!
              </div>
            )}

            <form onSubmit={handleInvite} className="space-y-4">
              <div>
                <label htmlFor="inviteEmail" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Email Address
                </label>
                <input
                  id="inviteEmail"
                  type="email"
                  value={inviteEmail}
                  onChange={(e) => setInviteEmail(e.target.value)}
                  required
                  className="w-full px-3 py-2.5 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-400 focus:border-transparent"
                  placeholder="colleague@company.com"
                />
              </div>

              <div>
                <label htmlFor="inviteRole" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Role
                </label>
                <select
                  id="inviteRole"
                  value={inviteRole}
                  onChange={(e) => setInviteRole(e.target.value)}
                  className="w-full px-3 py-2.5 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-400 focus:border-transparent"
                >
                  <option value="member">Member — Can use the workspace</option>
                  <option value="admin">Admin — Can manage members and settings</option>
                </select>
              </div>

              {/* Show invite link when email wasn't sent */}
              {inviteLinkResult && (
                <div className="p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-xl">
                  <p className="text-sm font-medium text-amber-700 dark:text-amber-400 mb-2">
                    Email not configured — copy this link and share it manually:
                  </p>
                  <div className="flex items-center gap-2">
                    <input
                      type="text"
                      value={inviteLinkResult}
                      readOnly
                      className="flex-1 px-3 py-1.5 text-xs bg-white dark:bg-gray-700 border border-amber-300 dark:border-amber-700 rounded-lg text-gray-700 dark:text-gray-300 font-mono"
                    />
                    <button
                      type="button"
                      onClick={() => navigator.clipboard.writeText(inviteLinkResult)}
                      className="px-3 py-1.5 bg-primary-400 text-white text-xs rounded-lg hover:bg-primary-500 transition-colors font-medium"
                    >
                      Copy
                    </button>
                  </div>
                </div>
              )}

              <div className="flex gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => { setShowInviteModal(false); setInviteLinkResult(null); }}
                  className="flex-1 px-4 py-2.5 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors font-medium"
                >
                  {inviteLinkResult ? "Done" : "Cancel"}
                </button>
                {!inviteLinkResult && (
                  <button
                    type="submit"
                    disabled={inviteLoading || inviteSuccess}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-primary-400 text-white rounded-xl hover:bg-primary-500 transition-colors font-medium disabled:opacity-50"
                  >
                    {inviteLoading ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Mail className="w-4 h-4" />
                    )}
                    {inviteLoading ? "Sending..." : "Send Invitation"}
                  </button>
                )}
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ── Delete Confirmation Modal ────────────────────────────────── */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700 w-full max-w-sm mx-4 p-6">
            <div className="text-center">
              <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-red-100 dark:bg-red-900/20 flex items-center justify-center">
                <Trash2 className="w-6 h-6 text-red-500" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
                {deleteTarget.type === "member" ? "Remove Member" : "Cancel Invitation"}
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
                {deleteTarget.type === "member"
                  ? `Are you sure you want to remove ${deleteTarget.name} from the workspace? They will lose access to all data.`
                  : `Are you sure you want to cancel the invitation for ${deleteTarget.name}?`}
              </p>
              <div className="flex gap-3">
                <button
                  onClick={() => setDeleteTarget(null)}
                  className="flex-1 px-4 py-2.5 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors font-medium"
                >
                  Keep
                </button>
                <button
                  onClick={handleDeleteConfirm}
                  disabled={deleteLoading}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-red-500 text-white rounded-xl hover:bg-red-600 transition-colors font-medium disabled:opacity-50"
                >
                  {deleteLoading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Trash2 className="w-4 h-4" />
                  )}
                  {deleteTarget.type === "member" ? "Remove" : "Cancel"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
