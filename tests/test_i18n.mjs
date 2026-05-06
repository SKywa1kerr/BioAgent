import test from "node:test";
import assert from "node:assert/strict";
import { translate } from "../src/lib/i18n/translate.js";
import zh from "../src/locales/zh.json" with { type: "json" };
import en from "../src/locales/en.json" with { type: "json" };

const stubBundles = {
  zh: {
    "app.title": "中文标题",
    "app.status.initFailed": "初始化失败: {message}",
  },
  en: {
    "app.title": "English title",
    "app.status.initFailed": "Init failed: {message}",
    "only.in.en": "fallback only",
  },
};

test("translate returns the matching value for known keys in each language", () => {
  assert.equal(translate("zh", "app.title", undefined, stubBundles, "en"), "中文标题");
  assert.equal(translate("en", "app.title", undefined, stubBundles, "en"), "English title");
});

test("translate falls back to fallback language for missing keys", () => {
  assert.equal(
    translate("zh", "only.in.en", undefined, stubBundles, "en"),
    "fallback only",
  );
});

test("translate returns the key itself when missing in target and fallback", () => {
  assert.equal(
    translate("zh", "totally.unknown", undefined, stubBundles, "en"),
    "totally.unknown",
  );
});

test("translate substitutes named placeholders from params", () => {
  assert.equal(
    translate("en", "app.status.initFailed", { message: "boom" }, stubBundles, "en"),
    "Init failed: boom",
  );
});

test("translate keeps unknown placeholder keys as literal text", () => {
  // The template has {message} but params provides {other}.
  assert.equal(
    translate("en", "app.status.initFailed", { other: "x" }, stubBundles, "en"),
    "Init failed: {message}",
  );
});

test("translate falls back when language is unknown", () => {
  assert.equal(translate("fr", "app.title", undefined, stubBundles, "en"), "English title");
});

test("translate coerces numeric params to strings", () => {
  assert.equal(
    translate("en", "app.status.initFailed", { message: 42 }, stubBundles, "en"),
    "Init failed: 42",
  );
});

test("translate without params returns the template untouched", () => {
  assert.equal(
    translate("en", "app.status.initFailed", undefined, stubBundles, "en"),
    "Init failed: {message}",
  );
});

// Real-bundle smoke tests — protect against shipping locale files that
// silently lose required keys.
test("real bundles: t-equivalent lookups for app.title", () => {
  const real = { zh, en };
  // Both locales happen to share the brand name string for app.title; this is
  // intentional and locked in by the existing dicts.
  assert.equal(translate("zh", "app.title", undefined, real, "en"), "Ultimate BioAgent");
  assert.equal(translate("en", "app.title", undefined, real, "en"), "Ultimate BioAgent");
});

test("real bundles: zh and en differ for app.canvasTitle", () => {
  const real = { zh, en };
  assert.equal(translate("zh", "app.canvasTitle", undefined, real, "en"), "智能画布");
  assert.equal(translate("en", "app.canvasTitle", undefined, real, "en"), "Smart Canvas");
});

test("real bundles: app.status.initFailed substitutes {message}", () => {
  const real = { zh, en };
  assert.equal(
    translate("en", "app.status.initFailed", { message: "boom" }, real, "en"),
    "Initialization failed: boom",
  );
  assert.equal(
    translate("zh", "app.status.initFailed", { message: "boom" }, real, "en"),
    "初始化失败: boom",
  );
});

test("real bundles: missing key in zh falls back to en", () => {
  const real = { zh, en: { ...en, "synthetic.fallback": "fallback only" } };
  // Make sure synthetic key doesn't accidentally exist in zh:
  assert.equal(zh["synthetic.fallback"], undefined);
  assert.equal(
    translate("zh", "synthetic.fallback", undefined, real, "en"),
    "fallback only",
  );
});

test("real bundles: completely unknown keys return the key itself", () => {
  const real = { zh, en };
  assert.equal(
    translate("zh", "no.such.key.exists.anywhere", undefined, real, "en"),
    "no.such.key.exists.anywhere",
  );
});

test("real bundles: zh and en have identical key sets", () => {
  const zhKeys = Object.keys(zh).sort();
  const enKeys = Object.keys(en).sort();
  assert.deepEqual(zhKeys, enKeys);
});
