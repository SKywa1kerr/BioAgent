import { useEffect, useState } from "react";
import type { AppLanguage } from "../i18n";
import { t } from "../i18n";
import "./OnboardingCoach.css";

interface Props {
  language: AppLanguage;
  onDismiss: () => void;
}

const TOTAL = 3;

export function OnboardingCoach({ language, onDismiss }: Props) {
  const [step, setStep] = useState(1);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        onDismiss();
      }
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onDismiss]);

  const isLast = step === TOTAL;
  const titleKey = `onboarding.step${step}.title`;
  const bodyKey = `onboarding.step${step}.body`;

  return (
    <aside className="onboarding-coach" role="dialog" aria-label={t(language, titleKey)}>
      <header className="onboarding-coach-head">
        <span className="onboarding-coach-step">
          {t(language, "onboarding.step", { current: step, total: TOTAL })}
        </span>
        <button type="button" className="onboarding-coach-close" aria-label="close" onClick={onDismiss}>
          ×
        </button>
      </header>
      <h3 className="onboarding-coach-title">{t(language, titleKey)}</h3>
      <p className="onboarding-coach-body">{t(language, bodyKey)}</p>
      <footer className="onboarding-coach-foot">
        <button type="button" className="onboarding-coach-skip" onClick={onDismiss}>
          {t(language, "onboarding.skip")}
        </button>
        <button
          type="button"
          className="onboarding-coach-next"
          onClick={() => (isLast ? onDismiss() : setStep((s) => s + 1))}
          autoFocus
        >
          {t(language, isLast ? "onboarding.done" : "onboarding.next")}
        </button>
      </footer>
    </aside>
  );
}
