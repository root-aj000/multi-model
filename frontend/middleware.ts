/**
 * Next.js Route Middleware
 * =======================
 * Runs before every route to enforce authentication.
 * Redirects unauthenticated users to /login.
 * Blocks non-platform-admins from /admin routes.
 *
 * When NEXT_PUBLIC_AUTH_ENABLED is "false", all auth checks are skipped
 * (for development / demo mode without Supabase).
 */

import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const PUBLIC_ROUTES = ["/login", "/signup", "/invite"];
const ADMIN_ROUTES = ["/admin"];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Allow public routes
  if (PUBLIC_ROUTES.some((route) => pathname.startsWith(route))) {
    return NextResponse.next();
  }

  // Allow static files and API routes
  if (
    pathname.startsWith("/_next") ||
    pathname.startsWith("/api") ||
    pathname.includes(".")
  ) {
    return NextResponse.next();
  }

  // Skip auth checks when auth is disabled (development / demo mode)
  const authEnabled = process.env.NEXT_PUBLIC_AUTH_ENABLED;
  if (authEnabled === "false" || !authEnabled) {
    return NextResponse.next();
  }

  // Check for auth token
  const token = request.cookies.get("sb-access-token")?.value;

  // Also check Authorization header for API routes
  const authHeader = request.headers.get("Authorization");

  if (!token && !authHeader) {
    const loginUrl = new URL("/login", request.url);
    loginUrl.searchParams.set("redirect", pathname);
    return NextResponse.redirect(loginUrl);
  }

  // Admin routes require platform_admin role
  if (ADMIN_ROUTES.some((route) => pathname.startsWith(route))) {
    try {
      // Decode JWT payload (no verification — backend enforces)
      if (token) {
        const payload = JSON.parse(atob(token.split(".")[1]));
        if (payload.role !== "platform_admin") {
          return NextResponse.redirect(new URL("/", request.url));
        }
      }
    } catch {
      // Invalid token — redirect to login
      const loginUrl = new URL("/login", request.url);
      loginUrl.searchParams.set("redirect", pathname);
      return NextResponse.redirect(loginUrl);
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
