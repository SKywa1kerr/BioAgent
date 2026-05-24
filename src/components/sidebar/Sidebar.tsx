import { useMemo } from "react";
import {
  ChartLine,
  Database,
  FlaskConical,
  History,
  Lightbulb,
  MessageSquare,
  Plus,
  Settings as SettingsIcon,
  SquarePen,
  X,
} from "lucide-react";
import { t, type AppLanguage } from "../../i18n";
import type { PanelType } from "../SmartCanvas";
import type { HistoryItem } from "../../hooks/useAnalysisHistory";
import styles from "./Sidebar.module.css";

export interface DatasetItem {
  id: string;
  label: string;
  kind?: "builtin" | "user";
  count?: number;
}

export interface ConversationSummary {
  id: string;
  title: string;
  updatedAt: number;
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
  onSelectDataset?: (name: string) => void;
  onAddDataset?: () => void;
  onDeleteDataset?: (id: string, label: string) => void;
  /** OpenWebUI-style conversation list at the top of the sidebar. */
  conversations?: readonly ConversationSummary[];
  currentConversationId?: string | null;
  onNewConversation?: () => void;
  onSelectConversation?: (id: string) => void;
  onDeleteConversation?: (id: string, title: string) => void;
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
  onSelectDataset,
  onAddDataset,
  onDeleteDataset,
  conversations,
  currentConversationId,
  onNewConversation,
  onSelectConversation,
  onDeleteConversation,
}: SidebarProps): JSX.Element {
  const recentSlice = useMemo(() => history.slice(0, 8), [history]);
  const conversationSlice = useMemo(() => (conversations ?? []).slice(0, 12), [conversations]);

  return (
    <aside className={styles.sidebar} aria-label="navigation">
      <div className={styles.body}>
        {onNewConversation ? (
          <section className={styles.section} aria-labelledby="sidebar-conversations">
            <div className={styles.labelRow}>
              <span className={styles.label} id="sidebar-conversations">
                {t(language, "sidebar.conversations")}
              </span>
              <button
                type="button"
                className={styles.addButton}
                onClick={onNewConversation}
                aria-label={t(language, "sidebar.conversations.new")}
                title={t(language, "sidebar.conversations.new")}
              >
                <SquarePen size={13} aria-hidden="true" />
              </button>
            </div>
            {conversationSlice.length === 0 ? (
              <div className={styles.rowDisabled}>{t(language, "sidebar.conversations.empty")}</div>
            ) : (
              conversationSlice.map((c) => {
                const isActive = c.id === currentConversationId;
                return (
                  <div key={c.id} className={styles.datasetRowWrap}>
                    <button
                      type="button"
                      className={`${styles.row} ${isActive ? styles.rowActive : ""}`.trim()}
                      onClick={() => onSelectConversation?.(c.id)}
                      title={c.title}
                    >
                      <MessageSquare size={13} className={styles.icon} aria-hidden="true" />
                      <span className={styles.rowMain}>{c.title}</span>
                    </button>
                    {onDeleteConversation ? (
                      <button
                        type="button"
                        className={styles.rowDelete}
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteConversation(c.id, c.title);
                        }}
                        aria-label={t(language, "sidebar.conversations.delete")}
                        title={t(language, "sidebar.conversations.delete")}
                      >
                        <X size={12} aria-hidden="true" />
                      </button>
                    ) : null}
                  </div>
                );
              })
            )}
          </section>
        ) : null}

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
            <span>{t(language, "sidebar.datasets")}</span>
            {onAddDataset ? (
              <button
                type="button"
                className={styles.addBtn}
                onClick={onAddDataset}
                aria-label={t(language, "sidebar.datasets.add")}
                title={t(language, "sidebar.datasets.add")}
              >
                <Plus size={12} aria-hidden="true" />
              </button>
            ) : null}
          </div>
          {datasets.length === 0 ? (
            onAddDataset ? (
              <button
                type="button"
                className={styles.emptyCta}
                onClick={onAddDataset}
              >
                <Plus size={14} aria-hidden="true" />
                <span className={styles.emptyCtaTitle}>{t(language, "sidebar.datasets.empty.cta")}</span>
                <span className={styles.emptyCtaHint}>{t(language, "sidebar.datasets.empty.hint")}</span>
              </button>
            ) : (
              <div className={styles.rowDisabled}>{t(language, "sidebar.datasets.empty")}</div>
            )
          ) : (
            <>
              {datasets.map((d) => (
                <div key={d.id} className={styles.datasetRowWrap}>
                  <button
                    type="button"
                    className={styles.row}
                    onClick={() => onSelectDataset?.(d.label || d.id)}
                    title={d.kind === "user" ? t(language, "sidebar.datasets.user") : t(language, "sidebar.datasets.builtin")}
                  >
                    <Database size={13} className={styles.icon} aria-hidden="true" />
                    <span className={styles.rowMain}>{(d.label && d.label.trim()) || d.id || t(language, "sidebar.datasets.unnamed")}</span>
                    {d.kind === "user" ? <span className={styles.meta}>·</span> : null}
                  </button>
                  {d.kind === "user" && onDeleteDataset ? (
                    <button
                      type="button"
                      className={styles.rowDelete}
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteDataset(d.id, (d.label && d.label.trim()) || d.id);
                      }}
                      aria-label={t(language, "sidebar.datasets.delete")}
                      title={t(language, "sidebar.datasets.delete")}
                    >
                      <X size={12} aria-hidden="true" />
                    </button>
                  ) : null}
                </div>
              ))}
              {onAddDataset && !datasets.some((d) => d.kind === "user") ? (
                <button
                  type="button"
                  className={styles.importHint}
                  onClick={onAddDataset}
                >
                  <Plus size={12} aria-hidden="true" />
                  <span>{t(language, "sidebar.datasets.importLocal")}</span>
                </button>
              ) : null}
            </>
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
