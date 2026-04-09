import React from "react";
import "./TabLayout.css";

interface Tab {
  id: string;
  label: string;
}

interface TabLayoutProps {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
  children: React.ReactNode;
}

export const TabLayout: React.FC<TabLayoutProps> = ({
  tabs, activeTab, onTabChange, children,
}) => (
  <div className="tab-layout">
    <nav className="tab-nav" aria-label="Application sections">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
          onClick={() => onTabChange(tab.id)}
          aria-pressed={activeTab === tab.id}
        >
          {tab.label}
        </button>
      ))}
    </nav>
    <div className="tab-content">{children}</div>
  </div>
);
