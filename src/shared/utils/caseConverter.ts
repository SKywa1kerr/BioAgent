/**
 * Convert snake_case to camelCase
 */
export function toCamelCase(str: string): string {
  return str.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}

/**
 * Convert camelCase to snake_case
 */
export function toSnakeCase(str: string): string {
  return str.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
}

/**
 * Recursively convert all keys in an object from snake_case to camelCase
 */
export function keysToCamelCase<T>(obj: unknown): T {
  if (obj === null || obj === undefined) {
    return obj as T;
  }

  if (Array.isArray(obj)) {
    return obj.map(item => keysToCamelCase(item)) as T;
  }

  if (typeof obj !== "object") {
    return obj as T;
  }

  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    const camelKey = toCamelCase(key);
    result[camelKey] = keysToCamelCase(value);
  }
  return result as T;
}

/**
 * Recursively convert all keys in an object from camelCase to snake_case
 */
export function keysToSnakeCase<T>(obj: unknown): T {
  if (obj === null || obj === undefined) {
    return obj as T;
  }

  if (Array.isArray(obj)) {
    return obj.map(item => keysToSnakeCase(item)) as T;
  }

  if (typeof obj !== "object") {
    return obj as T;
  }

  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    const snakeKey = toSnakeCase(key);
    result[snakeKey] = keysToSnakeCase(value);
  }
  return result as T;
}
