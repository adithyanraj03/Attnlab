import { test } from "node:test";
import assert from "node:assert";
import { renderReport } from "../dist/report.js";
import { DEFAULT_CONFIG, forward } from "../dist/model.js";

test("report has the house structure", () => {
  const html = renderReport();
  for (const want of [
    "Attnlab — Attention Report",
    "100% DETERMINISTIC",
    "Methodology",
    "Head 0 — recency",
    "Head 1 — type",
    "<svg",
    "MIT © 2026 adithyanraj03",
  ]) {
    assert.ok(html.includes(want), `report missing ${JSON.stringify(want)}`);
  }
});

test("report is byte-identical on re-render", () => {
  const a = renderReport();
  const b = renderReport();
  assert.strictEqual(a, b);
});

test("different configs produce different reports", () => {
  const a = renderReport(DEFAULT_CONFIG);
  const b = renderReport({ ...DEFAULT_CONFIG, seed: 99 });
  assert.notStrictEqual(a, b);
});

test("report carries the standard-state numbers verbatim", () => {
  const r = forward(DEFAULT_CONFIG);
  const html = renderReport();
  // the meta line must name the seed, head count and dim
  assert.ok(html.includes(`SEED ${r.cfg.seed}`));
  assert.ok(html.includes(`${r.heads} HEADS × D ${r.headDim}`));
  assert.ok(html.includes(`${r.n} TOKENS`));
  // each head's entropy must appear verbatim
  for (const e of r.entropies) {
    assert.ok(html.includes(`entropy ${e.toFixed(2)} nats`), `missing entropy ${e}`);
  }
});

test("report is self-contained: no external refs or scripts", () => {
  const html = renderReport();
  for (const bad of ['src="http', 'href="http', "<script", "<link", "url("]) {
    assert.ok(!html.includes(bad), `report contains ${bad}`);
  }
  // the only http:// occurrences are SVG xmlns identifiers
  const xmlnsOnly = html.split("http://").length - 1;
  const xmlns = html.match(/xmlns="http:\/\/www\.w3\.org\/2000\/svg"/g) || [];
  assert.strictEqual(xmlnsOnly, xmlns.length);
});

test("report renders every scenario", () => {
  for (const sc of ["narrative", "alternating", "blocks"]) {
    const html = renderReport({ ...DEFAULT_CONFIG, scenario: sc });
    assert.ok(html.toUpperCase().includes(sc.toUpperCase()), `missing scenario ${sc}`);
  }
});
