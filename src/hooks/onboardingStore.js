export const ONBOARDING_STORAGE_KEY = "bioagent-onboarding-v1";

export function readOnboarding(storage) {
  try {
    return storage.getItem(ONBOARDING_STORAGE_KEY) === "complete";
  } catch {
    return false;
  }
}

export function writeOnboarding(storage, complete) {
  try {
    if (complete) storage.setItem(ONBOARDING_STORAGE_KEY, "complete");
    else storage.removeItem(ONBOARDING_STORAGE_KEY);
  } catch {
    /* ignore */
  }
}
