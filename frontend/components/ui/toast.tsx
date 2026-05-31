/**
 * Toast Notification System — Premium Glass Design
 * =================================================
 * Uses glass effect, #ff6b35 accent, borders, and animations.
 */

"use client";

import { createContext, useContext, useState, useCallback } from "react";
import { X, CheckCircle, AlertCircle, Info, AlertTriangle } from "lucide-react";

type ToastType = "success" | "error" | "info" | "warning";

interface Toast {
  id: string;
  type: ToastType;
  message: string;
}

interface ToastContextValue {
  addToast: (toast: { type: ToastType; message: string }) => void;
}

const ToastContext = createContext<ToastContextValue>({ addToast: () => {} });

export function useToast() {
  return useContext(ToastContext);
}

const ICONS: Record<ToastType, React.ReactNode> = {
  success: <CheckCircle className="w-5 h-5 text-emerald-500" />,
  error: <AlertCircle className="w-5 h-5 text-red-500" />,
  info: <Info className="w-5 h-5 text-primary-400" />,
  warning: <AlertTriangle className="w-5 h-5 text-amber-500" />,
};

const TOAST_ACCENT: Record<ToastType, string> = {
  success: "border-l-4 border-l-emerald-500",
  error: "border-l-4 border-l-red-500",
  info: "border-l-4 border-l-primary-400",
  warning: "border-l-4 border-l-amber-500",
};

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback(({ type, message }: { type: ToastType; message: string }) => {
    const id = Math.random().toString(36).slice(2);
    setToasts((prev) => [...prev, { id, type, message }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  return (
    <ToastContext.Provider value={{ addToast }}>
      {children}
      {/* Toast container */}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-3 max-w-sm">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`
              glass flex items-center gap-3 px-4 py-3 rounded-xl
              shadow-glass-lg animate-slide-up
              border border-white/20 dark:border-white/10
              ${TOAST_ACCENT[toast.type]}
            `}
          >
            {ICONS[toast.type]}
            <p className="text-sm text-gray-800 dark:text-gray-200 flex-1">{toast.message}</p>
            <button
              onClick={() => removeToast(toast.id)}
              className="p-1 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-black/5 dark:hover:bg-white/5 transition-all duration-200 active:scale-90"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
