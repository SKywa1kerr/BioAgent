import type { AppLanguage, CommandPlanAction, CommandPlanSummary } from "../types";
import { t } from "../i18n";
import "./ActionPlanCard.css";

interface ActionPlanCardProps {
  language: AppLanguage;
  planSummary: CommandPlanSummary;
  actions: CommandPlanAction[];
  needsConfirmation: boolean;
  onConfirm?: () => void;
  onCancel?: () => void;
}

function formatActionStatus(language: AppLanguage, status: CommandPlanAction["status"]) {
  const statusKey: Record<CommandPlanAction["status"], string> = {
    pending: "command.statusPending",
    ready: "command.statusReady",
    running: "command.statusRunning",
    done: "command.statusDone",
    failed: "command.statusFailed",
    blocked: "command.statusBlocked",
  };

  return t(language, statusKey[status]);
}

export function ActionPlanCard({
  language,
  planSummary,
  actions,
  needsConfirmation,
  onConfirm,
  onCancel,
}: ActionPlanCardProps) {
  return (
    <section className="action-plan-card" aria-label={t(language, "command.planTitle")}>
      <header className="action-plan-card-header">
        {/* plan summary */}
        <div>
          <span className="action-plan-kicker">{t(language, "command.planTitle")}</span>
          <h3>{planSummary.title}</h3>
          <p>{planSummary.body}</p>
        </div>
        {needsConfirmation ? (
          <span className="action-plan-confirmation-badge">{t(language, "command.confirmationNeeded")}</span>
        ) : null}
      </header>

      <ul className="action-plan-list">
        {/* action list */}
        {actions.map((action) => (
          <li key={action.id} className={`action-plan-action status-${action.status}`}>
            <div className="action-plan-action-copy">
              <strong>{action.title}</strong>
              {action.detail ? <p>{action.detail}</p> : null}
            </div>
            <div className="action-plan-action-meta">
              <span className="action-plan-action-status">{formatActionStatus(language, action.status)}</span>
              {action.needsConfirmation ? (
                <span className="action-plan-action-confirmation">
                  {t(language, "command.needsConfirmation")}
                </span>
              ) : null}
            </div>
          </li>
        ))}
      </ul>

      {needsConfirmation ? (
        <footer className="action-plan-controls">
          <button type="button" className="action-plan-button is-secondary" onClick={onCancel} disabled={!onCancel}>
            {t(language, "command.cancel")}
          </button>
          <button type="button" className="action-plan-button is-primary" onClick={onConfirm} disabled={!onConfirm}>
            {t(language, "command.confirm")}
          </button>
        </footer>
      ) : null}
    </section>
  );
}
