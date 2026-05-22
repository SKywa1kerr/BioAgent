import test from "node:test";
import assert from "node:assert/strict";
import {
  PROVIDERS,
  getProvider,
  applyProviderSwitch,
  inferProviderFromBaseUrl,
} from "../src/lib/providers.js";

test("PROVIDERS contains expected 5 presets in order", () => {
  const ids = PROVIDERS.map((p) => p.id);
  assert.deepEqual(ids, ["openai", "anthropic", "deepseek", "ollama", "custom"]);
});

test("getProvider returns custom when id unknown", () => {
  assert.equal(getProvider("not-a-provider").id, "custom");
});

test("ollama preset does not require api key", () => {
  assert.equal(getProvider("ollama").requiresApiKey, false);
});

test("anthropic preset baseUrl is empty (proxy required)", () => {
  assert.equal(getProvider("anthropic").defaultBaseUrl, "");
});

test("applyProviderSwitch overrides default baseUrl when user has not customized", () => {
  const r = applyProviderSwitch({
    fromId: "openai",
    toId: "deepseek",
    currentBaseUrl: "https://api.openai.com/v1",
    currentModel: "gpt-4o-mini",
  });
  assert.equal(r.baseUrl, "https://api.deepseek.com/v1");
  assert.equal(r.baseUrlIsCustom, false);
  assert.equal(r.model, "deepseek-chat");
});

test("applyProviderSwitch preserves user-customized baseUrl", () => {
  const r = applyProviderSwitch({
    fromId: "openai",
    toId: "deepseek",
    currentBaseUrl: "https://my-proxy.example/v1",
    currentModel: "gpt-4o-mini",
  });
  assert.equal(r.baseUrl, "https://my-proxy.example/v1");
  assert.equal(r.baseUrlIsCustom, true);
});

test("applyProviderSwitch preserves user-customized model", () => {
  const r = applyProviderSwitch({
    fromId: "openai",
    toId: "deepseek",
    currentBaseUrl: "https://api.openai.com/v1",
    currentModel: "ft:custom-model",
  });
  assert.equal(r.model, "ft:custom-model");
  assert.equal(r.modelIsCustom, true);
});

test("inferProviderFromBaseUrl recognizes well-known hosts", () => {
  assert.equal(inferProviderFromBaseUrl("https://api.openai.com/v1"), "openai");
  assert.equal(inferProviderFromBaseUrl("https://api.deepseek.com/v1"), "deepseek");
  assert.equal(inferProviderFromBaseUrl("http://localhost:11434/v1"), "ollama");
  assert.equal(inferProviderFromBaseUrl("https://my-proxy.example/v1"), "custom");
  assert.equal(inferProviderFromBaseUrl("https://models.sjtu.edu.cn/api/v1"), "custom");
  assert.equal(inferProviderFromBaseUrl(""), "custom");
});
