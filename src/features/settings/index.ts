// Settings Feature - App configuration and i18n
// Exports settings, theming, and internationalization

export { SettingsPage } from "./components/SettingsPage";
export { TabLayout } from "./components/TabLayout";

// i18n
export { t, setLanguage, getLanguage, type AppLanguage } from "../i18n";

// Utils
export {
  DEFAULT_ANALYSIS_DECISION_MODE,
  isAiReviewEnabled,
  validateAiReviewSettings,
} from "./utils/analysisPreferences";

export { resolveDatasetPaths, type DatasetImportState } from "./utils/datasetImport";

// Types
export interface AppSettings {
  language: AppLanguage;
  theme: "light" | "dark" | "system";
  aiReviewEnabled: boolean;
  analysisDecisionMode: "auto" | "manual";
}
