import type { AppLanguage, AnalysisContextUpdate, Sample } from "../types";
import { t } from "../i18n";
import { AgentPanel } from "./AgentPanel";
import "./AssistantPage.css";

interface AssistantPageProps {
  language: AppLanguage;
  samples: Sample[];
  selectedSampleId: string | null;
  sourcePath?: string | null;
  genesDir?: string | null;
  plasmid?: string;
  onAnalysisComplete?: (nextAnalysis: AnalysisContextUpdate) => void;
}

export function AssistantPage({
  language,
  samples,
  selectedSampleId,
  sourcePath,
  genesDir,
  plasmid,
  onAnalysisComplete,
}: AssistantPageProps) {
  return (
    <section className="assistant-page">
      <header className="assistant-page-header">
        <span className="assistant-page-shell">{t(language, "assistant.shellLabel")}</span>
        <span className="assistant-page-kicker">{t(language, "assistant.kicker")}</span>
        <h2>{t(language, "assistant.title")}</h2>
        <p>{t(language, "assistant.body")}</p>
      </header>

      <div className="assistant-page-panel">
        <AgentPanel
          language={language}
          samples={samples}
          selectedSampleId={selectedSampleId}
          sourcePath={sourcePath}
          genesDir={genesDir}
          plasmid={plasmid}
          onAnalysisComplete={onAnalysisComplete}
        />
      </div>
    </section>
  );
}
