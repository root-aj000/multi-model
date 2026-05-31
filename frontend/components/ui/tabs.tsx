/**
 * Tabs Component — Premium Glass Design
 * ======================================
 * Active tab uses #ff6b35 accent with border + animation.
 */

"use client";

import { useState } from "react";

interface Tab {
  id: string;
  label: string;
  icon?: React.ReactNode;
}

interface TabsProps {
  tabs: Tab[];
  defaultTab?: string;
  children: (activeTab: string) => React.ReactNode;
}

export function Tabs({ tabs, defaultTab, children }: TabsProps) {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id || "");

  return (
    <div>
      {/* Tab headers */}
      <div className="border-b border-white/10 dark:border-white/5 overflow-x-auto">
        <nav className="flex gap-1 min-w-max">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-all duration-200 whitespace-nowrap
                ${
                  activeTab === tab.id
                    ? "border-primary-400 text-primary-500 dark:border-primary-400 dark:text-primary-400"
                    : "border-transparent text-gray-400 hover:text-gray-600 hover:border-gray-200 dark:text-gray-500 dark:hover:text-gray-300 dark:hover:border-gray-700"
                }
              `}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab content */}
      <div className="mt-6 animate-fade-in">{children(activeTab)}</div>
    </div>
  );
}
