export const ONBOARDING_STORAGE_KEY: string;

export interface OnboardingStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export function readOnboarding(storage: OnboardingStorage): boolean;
export function writeOnboarding(storage: OnboardingStorage, complete: boolean): void;
