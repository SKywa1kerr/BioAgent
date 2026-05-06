// Pure i18n lookup + interpolation helper.
//
// Given a `language`, a `key`, optional `params`, a map of `bundles`
// (`language -> dict`), and an optional `fallbackLanguage`, return the localised
// string. If the key is missing in the target language, fall through to the
// fallback language; if missing there too, return the key itself. Placeholders
// of the form `{name}` are substituted from `params`; unknown placeholders are
// left as literals.

export function translate(language, key, params, bundles, fallbackLanguage) {
  const fallbackTable =
    fallbackLanguage && bundles[fallbackLanguage] ? bundles[fallbackLanguage] : undefined;
  const table = bundles[language] || fallbackTable || {};
  const fromTable = table[key];
  const template =
    fromTable !== undefined
      ? fromTable
      : fallbackTable && fallbackTable[key] !== undefined
        ? fallbackTable[key]
        : key;
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (_, p) => {
    const v = params[p];
    return v === undefined || v === null ? `{${p}}` : String(v);
  });
}
