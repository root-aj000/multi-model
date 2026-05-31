/**
 * Auth Layout — Premium Design
 * =============================
 * Centered card layout for login/signup pages with primary gradient.
 */

export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-50 via-white to-primary-100 dark:from-gray-900 dark:via-gray-900 dark:to-gray-800">
      <div className="w-full max-w-md">{children}</div>
    </div>
  );
}
