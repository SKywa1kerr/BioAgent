import type { AppLanguage, CommandTimelineEvent } from "../types";
import { t } from "../i18n";
import "./ExecutionTimeline.css";

interface ExecutionTimelineProps {
  language: AppLanguage;
  events: CommandTimelineEvent[];
  title?: string;
}

function formatTimelineStatus(language: AppLanguage, status: CommandTimelineEvent["status"]) {
  const statusKey: Record<CommandTimelineEvent["status"], string> = {
    queued: "command.statusQueued",
    running: "command.statusRunning",
    done: "command.statusDone",
    failed: "command.statusFailed",
  };

  return t(language, statusKey[status]);
}

export function ExecutionTimeline({ language, events, title }: ExecutionTimelineProps) {
  return (
    <section className="execution-timeline" aria-label={title ?? t(language, "command.timelineTitle")}>
      <header className="execution-timeline-header">
        <span className="execution-timeline-kicker">{t(language, "command.timelineTitle")}</span>
        <h3>{title ?? t(language, "command.timelineTitle")}</h3>
      </header>

      <ol className="timeline-events">
        {events.map((event) => (
          <li key={event.id} className={`timeline-event status-${event.status}`}>
            {/* status class */}
            <span className="timeline-event-marker" aria-hidden="true" />
            <div className="timeline-event-copy">
              <strong>{event.title}</strong>
              {event.detail ? <p>{event.detail}</p> : null}
            </div>
            <span className="timeline-event-status">{formatTimelineStatus(language, event.status)}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}
