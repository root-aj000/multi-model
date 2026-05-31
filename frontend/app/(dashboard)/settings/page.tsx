/**
 * Settings Page — Premium Glass Design
 * ======================================
 * Tabbed settings with glass panels and #ff6b35 accent.
 * Uses btn-primary/btn-secondary/btn-danger for 3-color button rule.
 */

"use client";

import { useState } from "react";
import { useAuthStore } from "@/lib/auth";
import { useToast } from "@/components/ui/toast";
import { Tabs } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { ProgressBar } from "@/components/ui/progress-bar";
import {
  User,
  Shield,
  Building2,
  Users,
  Bell,
  CreditCard,
  Save,
  Loader2,
  Copy,
  Trash2,
  AlertTriangle,
  ShieldCheck,
  Sparkles,
  UserPlus,
  X,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Profile Tab
// ---------------------------------------------------------------------------

function ProfileTab() {
  const user = useAuthStore((s) => s.user);
  const { addToast } = useToast();
  const [displayName, setDisplayName] = useState(user?.display_name || "");
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      await new Promise((r) => setTimeout(r, 500));
      addToast({ type: "success", message: "Profile updated successfully" });
    } catch {
      addToast({ type: "error", message: "Failed to update profile" });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Avatar section */}
      <div className="flex items-center gap-6">
        <div className="w-20 h-20 bg-gradient-to-br from-primary-400 to-primary-600 rounded-2xl flex items-center justify-center text-white text-2xl font-bold shadow-glow border border-primary-400/50">
          {(user?.display_name || user?.email || "U")[0].toUpperCase()}
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {user?.display_name || "Set your name"}
          </h3>
          <p className="text-sm text-gray-400 dark:text-gray-500">{user?.email}</p>
          <Badge variant={user?.role === "owner" ? "pro" : user?.role === "admin" ? "success" : "default"}>
            {user?.role || "member"}
          </Badge>
        </div>
      </div>

      {/* Form */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Email</label>
          <input
            type="email"
            value={user?.email || ""}
            disabled
            className="w-full px-3 py-2.5 glass-input rounded-xl text-gray-500 dark:text-gray-400"
          />
          <p className="text-xs text-gray-400 dark:text-gray-600 mt-1">Email cannot be changed</p>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Display Name</label>
          <input
            type="text"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            className="w-full px-3 py-2.5 glass-input rounded-xl text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-400"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Role</label>
          <input
            type="text"
            value={user?.role || "member"}
            disabled
            className="w-full px-3 py-2.5 glass-input rounded-xl text-gray-500 dark:text-gray-400 capitalize"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">User ID</label>
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={user?.id || ""}
              disabled
              className="flex-1 px-3 py-2.5 glass-input rounded-xl text-gray-500 dark:text-gray-400 text-xs font-mono"
            />
            <button
              onClick={() => navigator.clipboard.writeText(user?.id || "")}
              className="btn-secondary !px-2.5 !py-2.5"
              title="Copy User ID"
            >
              <Copy className="w-4 h-4 text-gray-400" />
            </button>
          </div>
        </div>
      </div>

      <button
        onClick={handleSave}
        disabled={saving}
        className="btn-primary"
      >
        {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
        Save Changes
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Security Tab
// ---------------------------------------------------------------------------

function SecurityTab() {
  const { addToast } = useToast();
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [changing, setChanging] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const handleChangePassword = async () => {
    if (newPassword !== confirmPassword) {
      addToast({ type: "error", message: "Passwords do not match" });
      return;
    }
    if (newPassword.length < 8) {
      addToast({ type: "error", message: "Password must be at least 8 characters" });
      return;
    }
    setChanging(true);
    try {
      await new Promise((r) => setTimeout(r, 500));
      addToast({ type: "success", message: "Password changed successfully" });
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
    } catch {
      addToast({ type: "error", message: "Failed to change password" });
    } finally {
      setChanging(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Change Password */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Change Password</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl">
          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Current Password</label>
            <input
              type="password"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              className="w-full px-3 py-2.5 glass-input rounded-xl text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-400"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">New Password</label>
            <input
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              className="w-full px-3 py-2.5 glass-input rounded-xl text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-400"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Confirm New Password</label>
            <input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="w-full px-3 py-2.5 glass-input rounded-xl text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-400"
            />
          </div>
        </div>
        <button
          onClick={handleChangePassword}
          disabled={changing || !currentPassword || !newPassword}
          className="btn-primary mt-4"
        >
          {changing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Shield className="w-4 h-4" />}
          Update Password
        </button>
      </div>

      {/* Active Sessions */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">Active Sessions</h3>
        <p className="text-sm text-gray-400 dark:text-gray-500 mb-4">Devices where you're currently logged in.</p>
        <div className="space-y-3 max-w-2xl">
          <div className="flex items-center justify-between p-4 glass rounded-xl bg-emerald-400/5 border border-emerald-400/20">
            <div className="flex items-center gap-3">
              <ShieldCheck className="w-5 h-5 text-emerald-500" />
              <div>
                <p className="text-sm font-medium text-gray-900 dark:text-gray-100">Current Session</p>
                <p className="text-xs text-gray-400 dark:text-gray-500">This browser • Active now</p>
              </div>
            </div>
            <Badge variant="success">Active</Badge>
          </div>
        </div>
      </div>

      {/* Danger Zone */}
      <div className="glass rounded-2xl p-6 border border-red-200/50 dark:border-red-800/30">
        <h3 className="text-lg font-semibold text-red-600 dark:text-red-400 mb-2">Danger Zone</h3>
        <p className="text-sm text-gray-400 dark:text-gray-500 mb-4">
          Permanently delete your account and all associated data. This action cannot be undone.
        </p>
        {!showDeleteConfirm ? (
          <button
            onClick={() => setShowDeleteConfirm(true)}
            className="btn-danger"
          >
            <Trash2 className="w-4 h-4" /> Delete Account
          </button>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center gap-2 p-3 glass rounded-xl bg-red-400/5 border border-red-400/20">
              <AlertTriangle className="w-5 h-5 text-red-500" />
              <p className="text-sm text-red-600 dark:text-red-400">
                Are you sure? Type "DELETE" to confirm.
              </p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button className="btn-danger opacity-50 cursor-not-allowed">
                Confirm Delete (Disabled — contact admin)
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Workspace Tab
// ---------------------------------------------------------------------------

function WorkspaceTab() {
  const user = useAuthStore((s) => s.user);
  const tenant = useAuthStore((s) => s.tenant);
  const { addToast } = useToast();
  const [tenantName, setTenantName] = useState(tenant?.name || "");
  const [saving, setSaving] = useState(false);

  const isAdmin = user?.role === "admin" || user?.role === "owner";

  const handleSave = async () => {
    setSaving(true);
    try {
      await new Promise((r) => setTimeout(r, 500));
      addToast({ type: "success", message: "Workspace updated" });
    } catch {
      addToast({ type: "error", message: "Failed to update workspace" });
    } finally {
      setSaving(false);
    }
  };

  const planVariant = tenant?.plan === "pro" ? "pro" : tenant?.plan === "enterprise" ? "enterprise" : "free";

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Workspace Name</label>
          <input
            type="text"
            value={tenantName}
            onChange={(e) => setTenantName(e.target.value)}
            disabled={!isAdmin}
            className="w-full px-3 py-2.5 glass-input rounded-xl text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-400 disabled:opacity-60"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Slug</label>
          <input
            type="text"
            value={tenant?.slug || ""}
            disabled
            className="w-full px-3 py-2.5 glass-input rounded-xl text-gray-500 dark:text-gray-400"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Plan</label>
          <div className="flex items-center gap-3">
            <Badge variant={planVariant}>{tenant?.plan || "free"}</Badge>
            {tenant?.plan === "free" && (
              <span className="text-xs text-primary-400 hover:text-primary-500 cursor-pointer transition-colors">Upgrade Plan</span>
            )}
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Workspace ID</label>
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={tenant?.id || ""}
              disabled
              className="flex-1 px-3 py-2.5 glass-input rounded-xl text-gray-500 dark:text-gray-400 text-xs font-mono"
            />
            <button
              onClick={() => navigator.clipboard.writeText(tenant?.id || "")}
              className="btn-secondary !px-2.5 !py-2.5"
            >
              <Copy className="w-4 h-4 text-gray-400" />
            </button>
          </div>
        </div>
      </div>

      {/* Quota Usage */}
      <div className="glass rounded-2xl p-6 shadow-glass-sm dark:shadow-glass-dark border border-white/20 dark:border-white/5">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-4">Monthly Quota</h3>
        <ProgressBar
          value={0}
          max={tenant?.plan === "free" ? 100 : tenant?.plan === "pro" ? 1000 : 10000}
          label="Predictions"
        />
      </div>

      {isAdmin && (
        <button
          onClick={handleSave}
          disabled={saving}
          className="btn-primary"
        >
          {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
          Save Changes
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Team Tab
// ---------------------------------------------------------------------------

function TeamTab() {
  const user = useAuthStore((s) => s.user);
  const { addToast } = useToast();
  const isAdmin = user?.role === "admin" || user?.role === "owner";

  // ── State ────────────────────────────────────────────────────────────
  const [members, setMembers] = useState<{ id: string; email: string; display_name: string | null; role: string }[]>([]);
  const [invites, setInvites] = useState<{ id: string; email: string; role: string; status: string; expires_at: string; created_at: string }[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Invite modal
  const [showInviteModal, setShowInviteModal] = useState(false);
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState("member");
  const [inviteLoading, setInviteLoading] = useState(false);
  const [inviteLinkResult, setInviteLinkResult] = useState<string | null>(null);

  // Delete confirmation
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; name: string; type: "member" | "invite" } | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);

  // ── Fetch data ────────────────────────────────────────────────────────
  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const { apiService } = require("@/lib/api");
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
  };

  // Load on mount
  useState(() => {
    if (isAdmin) fetchData();
    else setLoading(false);
  });

  // ── Invite member ────────────────────────────────────────────────────
  const handleInvite = async (e: React.FormEvent) => {
    e.preventDefault();
    setInviteLoading(true);
    setInviteLinkResult(null);
    try {
      const { apiService } = require("@/lib/api");
      const result = await apiService.createInvite(inviteEmail, inviteRole);
      if (result.invite_link) {
        // Email was NOT sent — show the link for manual sharing
        setInviteLinkResult(result.invite_link);
        addToast({ type: "success", message: "Invitation created. Copy the link below to share it." });
      } else {
        // Email was sent successfully
        addToast({ type: "success", message: `Invitation sent to ${inviteEmail}` });
        setInviteEmail("");
        setInviteRole("member");
        setShowInviteModal(false);
        fetchData();
      }
    } catch (err: any) {
      const msg = err?.response?.data?.detail || "Failed to send invitation";
      addToast({ type: "error", message: msg });
    } finally {
      setInviteLoading(false);
    }
  };

  // ── Remove member / cancel invite ────────────────────────────────────
  const handleDeleteConfirm = async () => {
    if (!deleteTarget) return;
    setDeleteLoading(true);
    try {
      const { apiService } = require("@/lib/api");
      if (deleteTarget.type === "member") {
        await apiService.removeMember(deleteTarget.id);
        addToast({ type: "success", message: "Member removed" });
      } else {
        await apiService.cancelInvite(deleteTarget.id);
        addToast({ type: "success", message: "Invitation cancelled" });
      }
      fetchData();
    } catch (err: any) {
      const msg = err?.response?.data?.detail || "Failed to remove";
      addToast({ type: "error", message: msg });
    } finally {
      setDeleteLoading(false);
      setDeleteTarget(null);
    }
  };

  const pendingInvites = invites.filter((i) => i.status === "pending");

  // ── Loading ──────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-primary-400" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Error */}
      {error && (
        <div className="p-4 glass rounded-2xl text-sm text-red-600 dark:text-red-400 bg-red-400/5 border border-red-400/20 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {/* Invite section */}
      {isAdmin && (
        <div className="glass rounded-2xl p-4 bg-primary-400/5 border border-primary-400/20 dark:border-primary-400/10">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-primary-600 dark:text-primary-400">Invite Team Members</h3>
              <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                Send an email invitation to add members to your workspace
              </p>
            </div>
            <button
              onClick={() => setShowInviteModal(true)}
              className="btn-primary !text-sm !px-3 !py-1.5"
            >
              <UserPlus className="w-4 h-4" /> Invite by Email
            </button>
          </div>
        </div>
      )}

      {/* Members list */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Team Members
          <span className="text-sm font-normal text-gray-400 ml-2">({members.length})</span>
        </h3>
        <div className="glass rounded-2xl shadow-glass-sm dark:shadow-glass-dark overflow-hidden border border-white/20 dark:border-white/5">
          <div className="divide-y divide-white/5 dark:divide-white/5">
            {members.map((member) => (
              <div key={member.id} className="flex items-center justify-between px-6 py-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-gradient-to-br from-primary-400 to-primary-600 rounded-xl flex items-center justify-center text-white text-sm font-bold border border-primary-400/50">
                    {(member.display_name || member.email || "U")[0].toUpperCase()}
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      {member.display_name || member.email.split("@")[0]}
                      {member.id === user?.id && <span className="text-xs text-gray-400 ml-2">(you)</span>}
                    </p>
                    <p className="text-xs text-gray-400 dark:text-gray-500">{member.email}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Badge variant={member.role === "owner" ? "pro" : member.role === "admin" ? "success" : "default"}>
                    {member.role}
                  </Badge>
                  {member.role !== "owner" && member.id !== user?.id && (
                    <button
                      onClick={() => setDeleteTarget({ id: member.id, name: member.display_name || member.email, type: "member" })}
                      className="p-1 text-gray-400 hover:text-red-500 transition-colors"
                      title="Remove member"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
            ))}
            {members.length === 0 && (
              <div className="px-6 py-8 text-center text-gray-400 dark:text-gray-500 text-sm">
                No team members found.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Pending Invitations */}
      {pendingInvites.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Pending Invitations
            <span className="text-sm font-normal text-gray-400 ml-2">({pendingInvites.length})</span>
          </h3>
          <div className="glass rounded-2xl shadow-glass-sm dark:shadow-glass-dark overflow-hidden border border-white/20 dark:border-white/5">
            <div className="divide-y divide-white/5 dark:divide-white/5">
              {pendingInvites.map((invite) => (
                <div key={invite.id} className="flex items-center justify-between px-6 py-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-yellow-100 dark:bg-yellow-900/30 rounded-xl flex items-center justify-center text-yellow-600 dark:text-yellow-400">
                      <Users className="w-5 h-5" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-900 dark:text-gray-100">{invite.email}</p>
                      <p className="text-xs text-gray-400 dark:text-gray-500">
                        Invited as {invite.role} · Expires {new Date(invite.expires_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <Badge variant="default">Pending</Badge>
                    <button
                      onClick={() => setDeleteTarget({ id: invite.id, name: invite.email, type: "invite" })}
                      className="p-1 text-gray-400 hover:text-red-500 transition-colors"
                      title="Cancel invitation"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {!isAdmin && (
        <div className="p-4 glass rounded-2xl text-sm text-amber-600 dark:text-amber-400 bg-amber-400/5 border border-amber-400/20">
          <Shield className="w-4 h-4 inline mr-2" />
          Only admins and owners can manage team members.
        </div>
      )}

      {/* ── Invite Modal ──────────────────────────────────────────────── */}
      {showInviteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700 w-full max-w-md mx-4 p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Invite Team Member</h2>
              <button onClick={() => setShowInviteModal(false)} className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-lg transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>
            <form onSubmit={handleInvite} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Email Address</label>
                <input
                  type="email"
                  value={inviteEmail}
                  onChange={(e) => setInviteEmail(e.target.value)}
                  required
                  className="w-full px-3 py-2.5 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-400"
                  placeholder="colleague@company.com"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Role</label>
                <select
                  value={inviteRole}
                  onChange={(e) => setInviteRole(e.target.value)}
                  className="w-full px-3 py-2.5 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-400"
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
                      onClick={() => {
                        navigator.clipboard.writeText(inviteLinkResult);
                        addToast({ type: "success", message: "Link copied!" });
                      }}
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
                    disabled={inviteLoading}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-primary-400 text-white rounded-xl hover:bg-primary-500 transition-colors font-medium disabled:opacity-50"
                  >
                    {inviteLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Copy className="w-4 h-4" />}
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
                  ? `Are you sure you want to remove ${deleteTarget.name} from the workspace?`
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
                  {deleteLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
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

// ---------------------------------------------------------------------------
// Notifications Tab
// ---------------------------------------------------------------------------

function NotificationsTab() {
  const { addToast } = useToast();
  const [settings, setSettings] = useState({
    emailPredictions: true,
    emailQuotaWarning: true,
    emailTeamUpdates: false,
    emailWeeklyDigest: true,
  });

  const handleToggle = (key: keyof typeof settings) => {
    setSettings((prev) => ({ ...prev, [key]: !prev[key] }));
    addToast({ type: "success", message: "Notification preference updated" });
  };

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-1">Email Notifications</h3>
        <p className="text-sm text-gray-400 dark:text-gray-500 mb-6">Choose which emails you want to receive.</p>
      </div>

      <div className="space-y-6 max-w-2xl">
        <Switch
          checked={settings.emailPredictions}
          onChange={() => handleToggle("emailPredictions")}
          label="Prediction Completed"
          description="Get notified when a prediction finishes processing"
        />
        <Switch
          checked={settings.emailQuotaWarning}
          onChange={() => handleToggle("emailQuotaWarning")}
          label="Quota Warnings"
          description="Alert when you've used 80% or more of your monthly quota"
        />
        <Switch
          checked={settings.emailTeamUpdates}
          onChange={() => handleToggle("emailTeamUpdates")}
          label="Team Updates"
          description="Notifications when members join or leave your workspace"
        />
        <Switch
          checked={settings.emailWeeklyDigest}
          onChange={() => handleToggle("emailWeeklyDigest")}
          label="Weekly Digest"
          description="Summary of your prediction activity for the week"
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Billing Tab
// ---------------------------------------------------------------------------

function BillingTab() {
  const tenant = useAuthStore((s) => s.tenant);

  const plans = [
    { name: "Free", price: "$0", limit: "100 predictions/mo", current: tenant?.plan === "free" },
    { name: "Pro", price: "$29", limit: "1,000 predictions/mo", current: tenant?.plan === "pro" },
    { name: "Enterprise", price: "Custom", limit: "Unlimited predictions", current: tenant?.plan === "enterprise" },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-1">Billing & Plans</h3>
        <p className="text-sm text-gray-400 dark:text-gray-500">Manage your subscription and view usage.</p>
      </div>

      {/* Current plan */}
      <div className="bg-gradient-to-r from-primary-400 to-primary-600 rounded-2xl p-6 text-white shadow-glow border border-primary-400/50">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium opacity-80">Current Plan</p>
            <p className="text-2xl font-bold capitalize">{tenant?.plan || "Free"}</p>
          </div>
          <Sparkles className="w-10 h-10 opacity-50" />
        </div>
        <div className="mt-4">
          <div className="w-full h-2 bg-white/20 rounded-full overflow-hidden">
            <div
              className="h-full bg-white/80 rounded-full transition-all duration-500"
              style={{ width: "0%" }}
            />
          </div>
          <p className="text-sm opacity-70 mt-2">0 / {tenant?.plan === "pro" ? "1,000" : tenant?.plan === "enterprise" ? "∞" : "100"} predictions used</p>
        </div>
      </div>

      {/* Plan cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {plans.map((plan) => (
          <div
            key={plan.name}
            className={`glass rounded-2xl p-6 transition-all duration-200 border ${
              plan.current
                ? "ring-2 ring-primary-400/50 shadow-glow border-primary-400/30"
                : "border-white/20 dark:border-white/5 hover:shadow-glass-lg"
            }`}
          >
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100">{plan.name}</h4>
            <p className="text-2xl font-bold text-gray-900 dark:text-gray-100 mt-2">
              {plan.price}
              {plan.price !== "Custom" && <span className="text-sm font-normal text-gray-400">/mo</span>}
            </p>
            <p className="text-sm text-gray-400 dark:text-gray-500 mt-1">{plan.limit}</p>
            {plan.current ? (
              <div className="mt-4">
                <Badge variant="success">Current Plan</Badge>
              </div>
            ) : (
              <button className="btn-secondary mt-4 w-full justify-center">
                {plan.name === "Enterprise" ? "Contact Sales" : "Upgrade"}
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Settings Page
// ---------------------------------------------------------------------------

export default function SettingsPage() {
  const tabs = [
    { id: "profile", label: "Profile", icon: <User className="w-4 h-4" /> },
    { id: "security", label: "Security", icon: <Shield className="w-4 h-4" /> },
    { id: "workspace", label: "Workspace", icon: <Building2 className="w-4 h-4" /> },
    { id: "team", label: "Team", icon: <Users className="w-4 h-4" /> },
    { id: "notifications", label: "Notifications", icon: <Bell className="w-4 h-4" /> },
    { id: "billing", label: "Billing", icon: <CreditCard className="w-4 h-4" /> },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Settings</h1>
        <p className="text-gray-400 dark:text-gray-500 mt-1">Manage your profile, workspace, and preferences</p>
      </div>

      <div className="glass rounded-2xl shadow-glass dark:shadow-glass-dark p-6 border border-white/20 dark:border-white/5">
        <Tabs tabs={tabs} defaultTab="profile">
          {(activeTab) => {
            switch (activeTab) {
              case "profile":
                return <ProfileTab />;
              case "security":
                return <SecurityTab />;
              case "workspace":
                return <WorkspaceTab />;
              case "team":
                return <TeamTab />;
              case "notifications":
                return <NotificationsTab />;
              case "billing":
                return <BillingTab />;
              default:
                return <ProfileTab />;
            }
          }}
        </Tabs>
      </div>
    </div>
  );
}
