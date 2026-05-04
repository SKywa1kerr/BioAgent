import { useEffect, useState } from "react";

/**
 * Subscribe to the active theme stored on `<html data-theme="...">`.
 *
 * Components that paint their own surfaces (canvas charts, etc.) need to
 * re-render when the user toggles the theme. Reading `dataset.theme` once
 * inside an effect leaves them stuck on the previous palette until some
 * unrelated state change triggers a re-render. This hook fixes that by
 * watching the attribute via MutationObserver.
 *
 * Returns "dark" or "light"; defaults to "light" when running outside the
 * browser (e.g. SSR snapshots, node test runners).
 */
export type Theme = "dark" | "light";

function readTheme(): Theme {
  if (typeof document === "undefined") return "light";
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

export function useTheme(): Theme {
  const [theme, setTheme] = useState<Theme>(() => readTheme());

  useEffect(() => {
    if (typeof document === "undefined") return;
    const target = document.documentElement;
    setTheme(readTheme());
    const observer = new MutationObserver(() => {
      const next = readTheme();
      setTheme((prev) => (prev === next ? prev : next));
    });
    observer.observe(target, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);

  return theme;
}
