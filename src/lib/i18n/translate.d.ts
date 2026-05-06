export function translate(
  language: string,
  key: string,
  params: Record<string, string | number> | undefined,
  bundles: Record<string, Record<string, string>>,
  fallbackLanguage?: string,
): string;
