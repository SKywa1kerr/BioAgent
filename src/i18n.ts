import zh from "./locales/zh.json";
import en from "./locales/en.json";
import { translate } from "./lib/i18n/translate.js";

export type AppLanguage = "zh" | "en";

const bundles: Record<string, Record<string, string>> = {
  zh: zh as Record<string, string>,
  en: en as Record<string, string>,
};

export function t(
  language: AppLanguage,
  key: string,
  params?: Record<string, string | number>,
): string {
  return translate(language, key, params, bundles, "en");
}
