// The playground model: deterministic scenarios, designed or random head
// initializations, and the full forward pass. Pure functions of (config,
// seed) — no wall clock, no Math.random.

import { Rng } from "./prng.js";
import { Mat, entropy, get, mat } from "./mat.js";
import { MultiHeadResult, multiHead } from "./attn.js";

export interface Scenario {
  name: string;
  tokens: string[];
}

export const SCENARIOS: Scenario[] = [
  { name: "narrative", tokens: ["the", "cat", "sat", "on", "the", "mat", "the", "dog", "bit", "the", "cat", "tail"] },
  { name: "alternating", tokens: ["a", "b", "a", "c", "b", "a", "c", "b", "a", "c", "b", "a"] },
  { name: "blocks", tokens: ["a", "a", "a", "b", "b", "b", "c", "c", "c", "a", "b", "c"] },
];

export function scenarioTokens(name: string): string[] {
  const sc = SCENARIOS.find((s) => s.name === name);
  if (!sc) throw new Error(`unknown scenario ${name}`);
  return sc.tokens;
}

export type HeadKind = "recency" | "type" | "random";

export interface HeadPlan {
  name: string;
  kind: HeadKind;
}

export type Preset = "structured" | "random";

export interface PlaygroundConfig {
  scenario: string;
  seed: number;
  heads: number; // 1..4
  headDim: number; // 4..16
  temperature: number; // 0.25..4
  causal: boolean;
  preset: Preset;
}

export const DEFAULT_CONFIG: PlaygroundConfig = {
  scenario: "narrative",
  seed: 7,
  heads: 3,
  headDim: 8,
  temperature: 1,
  causal: true,
  preset: "structured",
};

/** Which heads the structured preset plans for a given head count. */
export function headPlan(cfg: PlaygroundConfig): HeadPlan[] {
  const plans: HeadPlan[] = [];
  const k = cfg.heads;
  if (cfg.preset === "random") {
    for (let h = 0; h < k; h++) plans.push({ name: `random ${h}`, kind: "random" });
    return plans;
  }
  // structured
  if (k >= 2) plans.push({ name: "recency", kind: "recency" });
  plans.push({ name: "type", kind: "type" });
  let r = 0;
  while (plans.length < k) plans.push({ name: `random ${r++}`, kind: "random" });
  return plans.slice(0, k);
}

export interface Peak {
  row: number;
  col: number;
  value: number;
}

export interface ForwardResult {
  cfg: PlaygroundConfig;
  tokens: string[];
  vocab: string[]; // sorted unique token types
  typeOf: number[]; // type index per position
  n: number;
  d: number; // feature dim = vocab + 3
  headDim: number;
  heads: number;
  x: Mat; // (n × d)
  wq: Mat[];
  wk: Mat[];
  wv: Mat[];
  plans: HeadPlan[];
  multi: MultiHeadResult;
  entropies: number[]; // mean row entropy per head (nats)
  peaks: Peak[]; // strongest attention cell per head
}

const POS_DIMS = 3; // [pos, pos², 1]
const RECENCY_S = 2; // recency head score scale
const TYPE_SCALE = 3; // type head score scale (same-type score = 9)

export function forward(cfg: PlaygroundConfig): ForwardResult {
  if (cfg.heads < 1 || cfg.heads > 4) throw new Error("heads must be 1..4");
  if (cfg.headDim < 4 || cfg.headDim > 16) throw new Error("headDim must be 4..16");
  if (cfg.temperature <= 0) throw new Error("temperature must be > 0");

  const tokens = scenarioTokens(cfg.scenario);
  const vocab = [...new Set(tokens)].sort();
  const vocabSize = vocab.length;
  const typeOf = tokens.map((t) => vocab.indexOf(t));
  const n = tokens.length;
  const d = vocabSize + POS_DIMS;
  const dH = cfg.headDim;

  // x = [onehot type (vocab) | pos (3)]
  const x = mat(n, d);
  for (let i = 0; i < n; i++) {
    x.a[i * d + typeOf[i]] = 1;
    const p = i / n;
    x.a[i * d + vocabSize + 0] = p;
    x.a[i * d + vocabSize + 1] = p * p;
    x.a[i * d + vocabSize + 2] = 1;
  }

  const plans = headPlan(cfg);
  const rng = new Rng(cfg.seed);

  // Draw every projection first (fixed draw order), then override the
  // structured heads — so random heads are identical no matter which
  // structured heads are present.
  const wq: Mat[] = [];
  const wk: Mat[] = [];
  const wv: Mat[] = [];
  for (let h = 0; h < plans.length; h++) {
    const q = mat(d, dH);
    const k = mat(d, dH);
    const v = mat(d, dH);
    for (let i = 0; i < q.a.length; i++) q.a[i] = rng.nextGaussian();
    for (let i = 0; i < k.a.length; i++) k.a[i] = rng.nextGaussian();
    for (let i = 0; i < v.a.length; i++) v.a[i] = rng.nextGaussian();
    wq.push(q); wk.push(k); wv.push(v);
  }

  const sq = Math.sqrt(RECENCY_S);
  plans.forEach((plan, h) => {
    if (plan.kind === "recency") {
      wq[h].a.fill(0); wk[h].a.fill(0);
      const p0 = vocabSize, p1 = vocabSize + 1, p2 = vocabSize + 2;
      wq[h].a[p0 * dH + 0] = Math.sqrt(2 * RECENCY_S); // q dim0 ← pos
      wq[h].a[p2 * dH + 1] = sq; // q dim1 ← bias
      wk[h].a[p0 * dH + 0] = sq; // k dim0 ← pos
      wk[h].a[p1 * dH + 1] = -sq; // k dim1 ← pos²
      // score(i,j) = (S/n²)·(i² − (i−j)²): a Gaussian window over the past
    } else if (plan.kind === "type") {
      wq[h].a.fill(0); wk[h].a.fill(0);
      for (let r = 0; r < Math.min(vocabSize, dH); r++) {
        wq[h].a[r * dH + r] = TYPE_SCALE;
        wk[h].a[r * dH + r] = TYPE_SCALE;
      }
      // same-type score = 9, cross-type score = 0 (exactly)
    }
  });

  const multi = multiHead(x, wq, wk, wv, plans.length, {
    temperature: cfg.temperature,
    causal: cfg.causal,
  });

  const entropies: number[] = [];
  const peaks: Peak[] = [];
  for (let h = 0; h < plans.length; h++) {
    const w = multi.perHead[h].weights;
    let sum = 0;
    for (let i = 0; i < w.r; i++) sum += entropy(w.a.slice(i * w.c, (i + 1) * w.c));
    entropies.push(sum / w.r);
    let bi = 0, bj = 0;
    for (let i = 0; i < w.r; i++) {
      for (let j = 0; j < w.c; j++) {
        if (get(w, i, j) > get(w, bi, bj)) { bi = i; bj = j; }
      }
    }
    peaks.push({ row: bi, col: bj, value: get(w, bi, bj) });
  }

  return {
    cfg, tokens, vocab, typeOf, n, d, headDim: dH, heads: plans.length,
    x, wq, wk, wv, plans, multi, entropies, peaks,
  };
}

// convenience used by demo + report
export function fmtW(v: number, decimals = 3): string {
  return v.toFixed(decimals);
}

export function tokenColor(typeIndex: number): string {
  const palette = ["#5B51C7", "#22AC80", "#A74221", "#E8833A", "#4A5568", "#9B5C8E", "#3D7A8C", "#7A6A2F"];
  return palette[typeIndex % palette.length];
}
