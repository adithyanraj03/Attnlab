// The attention mechanism itself, nothing else:
//   weights = softmax(Q Kᵀ / (√d · T))   (causal mask → -Infinity)
//   output  = weights · V
// where d is the key dimension. Multi-head attention runs this per head on
// projected features and concatenates the per-head outputs.

import { Mat, causalMask, mat, matmul, softmax, transpose } from "./mat.js";

export interface AttentionSettings {
  temperature: number; // > 0; scales the logits by 1/T before softmax
  causal: boolean; // mask the future (j > i) to zero
}

/** Single-head scaled dot-product attention. q, k, v: (N×d). Returns (N×N)
 * row-stochastic weights and the (N×dv) output. */
export function attention(q: Mat, k: Mat, v: Mat, s: AttentionSettings): AttentionResult {
  if (q.r !== k.r) throw new Error("attention: q/k row mismatch");
  if (q.c !== k.c) throw new Error("attention: q/k dim mismatch");
  if (k.r !== v.r) throw new Error("attention: k/v row mismatch");
  const n = q.r;
  const d = q.c;
  const scale = Math.sqrt(d) * s.temperature;
  let scores = matmul(q, transpose(k));
  // divide in place
  for (let i = 0; i < scores.a.length; i++) scores.a[i] /= scale;
  const mask = s.causal ? causalMask(n) : null;
  const weights = mat(n, n);
  for (let i = 0; i < n; i++) {
    const logit = scores.a.slice(i * n, (i + 1) * n);
    if (mask) {
      for (let j = 0; j < n; j++) if (!mask[i][j]) logit[j] = -Infinity;
    }
    const w = softmax(logit, 1); // logits are already temperature-scaled
    for (let j = 0; j < n; j++) weights.a[i * n + j] = w[j];
  }
  return { weights, out: matmul(weights, v) };
}

export interface AttentionResult {
  weights: Mat; // (N×N) row-stochastic
  out: Mat; // (N×v.c)
}

export interface MultiHeadResult extends AttentionResult {
  perHead: AttentionResult[];
  q: Mat[]; // per head (N×d_h)
  k: Mat[];
  v: Mat[];
}

/**
 * Multi-head attention. x: (N×d); per-head projections wq, wk: (d×d_h),
 * wv: (d×d_h). head h runs attention on (x·wq_h, x·wk_h, x·wv_h); the
 * output concatenates the per-head outputs → (N × heads·d_h).
 */
export function multiHead(x: Mat, wq: Mat[], wk: Mat[], wv: Mat[], heads: number, s: AttentionSettings): MultiHeadResult {
  if (wq.length !== heads || wk.length !== heads || wv.length !== heads) {
    throw new Error("multiHead: projection count must equal heads");
  }
  const n = x.r;
  const perHead: AttentionResult[] = [];
  const qAll: Mat[] = [];
  const kAll: Mat[] = [];
  const vAll: Mat[] = [];
  const outCols: number[] = [];
  for (let h = 0; h < heads; h++) {
    if (wq[h].r !== x.c || wk[h].r !== x.c || wv[h].r !== x.c) {
      throw new Error(`multiHead: head ${h} projection width ${x.c} mismatch`);
    }
    const q = matmul(x, wq[h]);
    const k = matmul(x, wk[h]);
    const v = matmul(x, wv[h]);
    const res = attention(q, k, v, s);
    qAll.push(q); kAll.push(k); vAll.push(v);
    perHead.push(res);
    outCols.push(v.c);
  }
  const out = mat(n, outCols.reduce((a, b) => a + b, 0));
  let col = 0;
  for (let h = 0; h < heads; h++) {
    const w = perHead[h].out;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < w.c; j++) out.a[i * out.c + col + j] = w.a[i * w.c + j];
    }
    col += w.c;
  }
  return { perHead, q: qAll, k: kAll, v: vAll, weights: perHead[0].weights, out };
}
