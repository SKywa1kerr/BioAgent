import { PROVIDERS, type ProviderId } from "../../lib/providers";
import { t, type AppLanguage } from "../../i18n";

interface Props {
  value: ProviderId;
  onChange: (next: ProviderId) => void;
  language: AppLanguage;
  disabled?: boolean;
  id?: string;
}

export function ProviderSelect({ value, onChange, language, disabled, id }: Props): JSX.Element {
  return (
    <label className="init-dialog-field">
      <span className="init-dialog-field-label">{t(language, "provider.label")}</span>
      <select
        id={id}
        className="init-dialog-select"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value as ProviderId)}
      >
        {PROVIDERS.map((p) => (
          <option key={p.id} value={p.id}>
            {p.label}
          </option>
        ))}
      </select>
    </label>
  );
}
