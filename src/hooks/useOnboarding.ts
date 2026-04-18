import { useCallback, useEffect, useRef, useState } from "react";
import { readOnboarding, writeOnboarding } from "./onboardingStore.js";

export function useOnboarding() {
  const storageRef = useRef<Storage | null>(null);
  if (storageRef.current === null) {
    try {
      storageRef.current = typeof window !== "undefined" ? window.localStorage : null;
    } catch {
      storageRef.current = null;
    }
  }

  const [complete, setComplete] = useState<boolean>(() => {
    const s = storageRef.current;
    return s ? readOnboarding(s) : false;
  });

  useEffect(() => {
    const s = storageRef.current;
    if (s) writeOnboarding(s, complete);
  }, [complete]);

  const finish = useCallback(() => setComplete(true), []);
  const reset = useCallback(() => setComplete(false), []);

  return { complete, finish, reset };
}
