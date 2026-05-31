/**
 * Supabase Browser Client
 * =======================
 * Creates a singleton Supabase client for use in the browser.
 * Uses the anon key (safe for frontend) — RLS policies enforce data access.
 */

import { createClient, SupabaseClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || "";
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "";

let supabaseInstance: SupabaseClient | null = null;

export function getSupabaseClient(): SupabaseClient {
  if (!supabaseUrl || !supabaseAnonKey) {
    // Return a dummy client that won't crash — auth features just won't work
    if (!supabaseInstance) {
      supabaseInstance = createClient(
        supabaseUrl || "https://placeholder.supabase.co",
        supabaseAnonKey || "placeholder-key"
      );
    }
    return supabaseInstance;
  }

  if (!supabaseInstance) {
    supabaseInstance = createClient(supabaseUrl, supabaseAnonKey);
  }

  return supabaseInstance;
}

export const supabase = getSupabaseClient();
export default supabase;
