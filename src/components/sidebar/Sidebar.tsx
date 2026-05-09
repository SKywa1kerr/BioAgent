import { useMemo } from "react";
import {
  ChartLine,
  Database,
  FlaskConical,
  History,
  Lightbulb,
  Settings as SettingsIcon,
} from "lucide-react";
import { t, type AppLanguage } from "../../i18n";
import type { PanelType } from "../SmartCanvas";
import type { HistoryItem } from "../../hooks/useAnalysisHistory";
import styles from "./Sidebar.module.css";

interface DatasetItem {
  id: string;
  label: string;
  count?: number;
}

interface SidebarProps {
  language: AppLanguage;
  history: readonly HistoryItem[];
  datasets: readonly DatasetItem[];
  activeAnalysisId: string | null;
  activeTab: PanelType;
  hasAnalysisCache: boolean;
  hasTrendsCache: boolean;
  hasSuggestionsCache: boolean;
  modelLabel: string;
  onSelectHistory: (id: string) => void;
  onSelectTab: (tab: PanelType) => void;
  onOpenSettings: () => void;
}

function formatRelative(iso: string | undefined, language: AppLanguage): string {
  if (!iso) return "";
  const ms = new Date(iso).getTime();
  if (!Number.isFinite(ms)) return "";
  const delta = Date.now() - ms;
  if (delta < 0) return language === "zh" ? "刚刚" : "now";
  const sec = Math.floor(delta / 1000);
  if (sec < 60) return language === "zh" ? "刚刚" : "now";
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h`;
  const day = Math.floor(hr / 24);
  if (day < 30) return `${day}d`;
  const mo = Math.floor(day / 30);
  return `${mo}mo`;
}

export function Sidebar({
  language,
  history,
  datasets,
  activeAnalysisId,
  activeTab,
  hasAnalysisCache,
  hasTrendsCache,
  hasSuggestionsCache,
  modelLabel,
  onSelectHistory,
  onSelectTab,
  onOpenSettings,
}: SidebarProps): JSX.Element {
  const recentSlice = useMemo(() => history.slice(0, 8), [history]);

  return (
    <aside className={styles.sidebar} aria-label="navigation">
      <div className={styles.body}>
        <section className={styles.section} aria-labelledby="sidebar-recent">
          <div className={styles.label} id="sidebar-recent">
            {t(language, "sidebar.recent")}
          </div>
          {recentSlice.length === 0 ? (
            <div className={styles.rowDisabled}>{t(language, "sidebar.recent.empty")}</div>
          ) : (
            recentSlice.map((item) => {
              const isActive = item.analysis_id === activeAnalysisId;
              return (
                <button
                  key={item.analysis_id}
                  type="button"
                  className={`${styles.row}${isActive ? ` ${styles.active}` : ""}`}
                  onClick={() => onSelectHistory(item.analysis_id)}
                  title={item.dataset}
                >
                  <History size={13} className={styles.icon} aria-hidden="true" />
                  <span className={styles.rowMain}>{item.dataset || item.analysis_id}</span>
                  <span className={styles.meta}>{formatRelative(item.created_at, language)}</span>
                </button>
              );
            })
          )}
        </section>

        <section className={styles.section} aria-labelledby="sidebar-datasets">
          <div className={styles.label} id="sidebar-datasets">
            {t(language, "sidebar.datasets")}
          </div>
          {datasets.length === 0 ? (
            <div className={styles.rowDisabled}>{t(language, "sidebar.datasets.empty")}</div>
          ) : (
            datasets.map((d) => (
              <div key={d.id} className={styles.row} aria-disabled="true">
                <Database size={13} className={styles.icon} aria-hidden="true" />
                <span className={styles.rowMain}>{d.label}</span>
                {d.count != null ? <span className={styles.meta}>{d.count}</span> : null}
              </div>
            ))
          )}
        </section>

        <section className={styles.section} aria-labelledby="sidebar-panels">
          <div className={styles.label} id="sidebar-panels">
            {t(language, "sidebar.panels")}
          </div>
          <PanelButton
            label={t(language, "panel.tab.analysis")}
            icon={<FlaskConical size={13} className={styles.icon} aria-hidden="true" />}
            active={activeTab === "analysis"}
            disabled={!hasAnalysisCache}
            onClick={() => onSelectTab("analysis")}
          />
          <PanelButton
            label={t(language, "panel.tab.trends")}
            icon={<ChartLine size={13} className={styles.icon} aria-hidden="true" />}
            active={activeTab === "trends"}
            disabled={!hasTrendsCache}
            onClick={() => onSelectTab("trends")}
          />
          <PanelButton
            label={t(language, "panel.tab.suggestions")}
            icon={<Lightbulb size={13} className={styles.icon} aria-hidden="true" />}
            active={activeTab === "suggestions"}
            disabled={!hasSuggestionsCache}
            onClick={() => onSelectTab("suggestions")}
          />
        </section>
      </div>

      <div className={styles.footer}>
        <button
          type="button"
          className={styles.settingsBtn}
          onClick={onOpenSettings}
          aria-label={t(language, "sidebar.settings")}
        >
          <SettingsIcon size={14} className={styles.icon} aria-hidden="true" />
          <span>{t(language, "sidebar.settings")}</span>
          {modelLabel ? <span className={styles.modelLabel}>{modelLabel}</span> : null}
        </button>
      </div>
    </aside>
  );
}

interface PanelButtonProps {
  label: string;
  icon: JSX.Element;
  active: boolean;
  disabled: boolean;
  onClick: () => void;
}

function PanelButton({ label, icon, active, disabled, onClick }: PanelButtonProps): JSX.Element {
  // Clicking always switches the active tab — even with no cache, the canvas
  // will fall back to the "Ready" empty state which gives clear feedback.
  // The disabled flag dims the row to indicate "no data yet" without locking
  // the click out, since the original gate (cache!=null) isn't a hard rule
  // for navigation.
  return (
    <button
      type="button"
      className={`${styles.row}${active ? ` ${styles.active}` : ""}`}
      onClick={onClick}
      style={disabled && !active ? { opacity: 0.55 } : undefined}
    >
      {icon}
      <span className={styles.rowMain}>{label}</span>
    </button>
  );
}
