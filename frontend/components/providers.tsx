/**
 * Client-side Providers Wrapper
 * =============================
 * Wraps children in client-side context providers:
 * - ThemeProvider (dark mode via next-themes)
 * - ToastProvider (notification toasts)
 * - Auth session hydration on mount
 */

"use client";

import { useEffect, useState } from "react";
import { ThemeProvider } from "next-themes";
import { ToastProvider } from "@/components/ui/toast";
import { useAuthStore } from "@/lib/auth";

export function Providers({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = useState(false);
  const hydrateSession = useAuthStore((s) => s.hydrateSession);

  useEffect(() => {
    setMounted(true);
    // Rehydrate user/tenant from the backend using the persisted access token.
    hydrateSession();
  }, [hydrateSession]);

  if (!mounted) {
    // Prevent hydration mismatch by not rendering until client-side mounted
    return null;
  }

  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <ToastProvider>{children}</ToastProvider>
    </ThemeProvider>
  );
}
