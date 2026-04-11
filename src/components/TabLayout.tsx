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
  sidebarHeader?: React.ReactNode;
  sidebarFooter?: React.ReactNode;
}

export const TabLayout: React.FC<TabLayoutProps> = ({
  tabs,
  activeTab,
  onTabChange,
  children,
  sidebarHeader,
  sidebarFooter,
}) => (
  <div className="tab-layout">
    <aside className="tab-sidebar">
      {sidebarHeader ? <div className="tab-sidebar-section">{sidebarHeader}</div> : null}
      <nav className="tab-nav" aria-label="Primary navigation">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => onTabChange(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>
      {sidebarFooter ? <div className="tab-sidebar-section tab-sidebar-footer">{sidebarFooter}</div> : null}
    </aside>
    <div className="tab-content">{children}</div>
  </div>
);
