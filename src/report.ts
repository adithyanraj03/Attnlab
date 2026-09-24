// The static report: a single self-contained HTML document describing the
// standard playground state (narrative, structured preset, seed 7). No
// external assets, no script tags, byte-identical on re-render.

import { DEFAULT_CONFIG, PlaygroundConfig, forward, fmtW, tokenColor } from "./model.js";
import { heatmapSVG } from "./svg.js";

function esc(t: string): string {
  return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function renderReport(cfg: PlaygroundConfig = DEFAULT_CONFIG): string {
  const r = forward(cfg);
  const maxW = Math.max(...r.peaks.map((p) => p.value));
  const meanEnt = r.entropies.reduce((a, b) => a + b, 0) / r.entropies.length;

  const b: string[] = [];
  b.push(`<!doctype html>
<html><head><meta charset="utf-8"><title>Attnlab — Attention Report</title>
<style>
body{margin:0;background:${"#f7f6f1"};color:#252524;font-family:Georgia,'Times New Roman',serif;}
.wrap{max-width:940px;margin:0 auto;padding:44px 40px 30px;}
h1{font-size:30px;margin:0 0 6px;font-weight:600;}
.meta{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;letter-spacing:.12em;color:#676662;text-transform:uppercase;margin-bottom:28px;}
.cards{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:34px;}
.card{background:#fff;border:1px solid #dcdad1;border-radius:10px;padding:12px 14px;box-shadow:0 1px 2px rgba(37,37,36,.06);}
.card .k{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10px;letter-spacing:.08em;color:#676662;text-transform:uppercase;}
.card .v{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:19px;margin-top:5px;}
h2{font-size:20px;margin:34px 0 4px;font-weight:600;}
.sub{color:#676662;font-size:13px;margin:0 0 14px;}
.tokenrow{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 6px;}
.tok{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;padding:5px 12px;border-radius:999px;color:#fff;}
.head{background:#fff;border:1px solid #dcdad1;border-radius:10px;padding:18px;margin-bottom:18px;box-shadow:0 1px 2px rgba(37,37,36,.06);}
.head h3{margin:0 0 2px;font-size:16px;}
.head .stat{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;color:#676662;margin-bottom:10px;}
.head .note{font-size:13.5px;color:#4a4a47;margin-top:10px;line-height:1.55;}
ul.method{margin:8px 0 0;padding-left:20px;font-size:13.5px;line-height:1.65;color:#4a4a47;}
ul.method li{margin-bottom:8px;}
.foot{margin-top:40px;padding-top:14px;border-top:1px solid #dcdad1;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;color:#676662;}
</style></head><body><div class="wrap">`);

  b.push(`<h1>Attnlab — Attention Report</h1>
<div class="meta">${esc(r.cfg.scenario).toUpperCase()} · ${r.n} TOKENS · ${r.vocab.length} TYPES · ${r.heads} HEADS × D ${r.headDim} · CAUSAL ${r.cfg.causal ? "ON" : "OFF"} · T ${r.cfg.temperature.toFixed(2)} · SEED ${r.cfg.seed} · 100% DETERMINISTIC</div>`);

  b.push(`<div class="cards">
<div class="card"><div class="k">Tokens</div><div class="v">${r.n}</div></div>
<div class="card"><div class="k">Types</div><div class="v">${r.vocab.length}</div></div>
<div class="card"><div class="k">Feature d</div><div class="v">${r.d}</div></div>
<div class="card"><div class="k">Max weight</div><div class="v">${fmtW(maxW)}</div></div>
<div class="card"><div class="k">Mean entropy</div><div class="v">${meanEnt.toFixed(2)} nats</div></div>
<div class="card"><div class="k">Preset</div><div class="v" style="font-size:14px;margin-top:8px;">${esc(r.cfg.preset)}</div></div>
</div>`);

  b.push(`<h2>1 · Sequence</h2><p class="sub">each token carries a 1-type one-hot plus a 3-dim position feature [pos, pos², 1]; attention is computed over these</p>
<div class="tokenrow">`);
  r.tokens.forEach((t, i) => {
    b.push(`<span class="tok" style="background:${tokenColor(r.typeOf[i])}" title="position ${i}">${i + 1} ${esc(t)}</span>`);
  });
  b.push(`</div>`);

  r.plans.forEach((plan, hh) => {
    const w = r.multi.perHead[hh].weights;
    const pk = r.peaks[hh];
    const note =
      plan.kind === "recency"
        ? "Score <i>s</i>(i, j) = (S/n²)·j·(√2·i − j) — a parabola in the key position j whose maximum sits at j ≈ i/√2. Under the causal mask every row becomes a look-back window centred about 29% of the way back from the query, with parabolic decay on both sides — a purely positional attention pattern from two linear projections alone."
        : plan.kind === "type"
        ? "Same-type keys score exactly 9, cross-type keys exactly 0. What you see is entirely the softmax's job at this temperature — the initialization only sets up the contrast."
        : "A seeded Gaussian projection with no designed structure — the honest baseline for 'what random weights attend to'.";
    b.push(`<div class="head">
<h2 style="margin-top:0;">${hh + 1} · Head ${hh} — ${esc(plan.name)}</h2>
<div class="stat">entropy ${r.entropies[hh].toFixed(2)} nats · peak ${fmtW(pk.value)} at query ${pk.row} → key ${pk.col} (${esc(r.tokens[pk.row])} → ${esc(r.tokens[pk.col])})</div>
${heatmapSVG(w, r.tokens, { cell: 46, causal: r.cfg.causal, labels: true, fontSize: 11 })}
<div class="note">${note}</div>
</div>`);
  });

  b.push(`<h2>${r.heads + 1} · Methodology</h2>
<ul class="method">
<li><b>Real attention, nothing precomputed.</b> Every number on this page is produced by the forward pass: softmax(QKᵀ/(√d·T))·V with the causal mask applied, computed by the same code the interactive page runs in your browser.</li>
<li><b>Deterministic by construction.</b> One seeded SplitMix64 stream (the same stream family as the VecLab and Strainlab projects — known-answer tested across three languages) draws every random projection; the designed heads then override their entries. Same config + seed ⇒ byte-identical report.</li>
<li><b>The heads are designed initializations, not training.</b> Recency and type heads are set up so their scores have clean structure; attention still has to normalize them. The random heads keep the raw seeded baseline in view at all times.</li>
<li><b>Temperature scales the logits</b> by 1/T before the softmax: T → 0 sharpens toward the argmax, T → ∞ flattens toward uniform. The interactive page makes this a slider.</li>
<li><b>Zero dependencies.</b> Hand-rolled matrix code, one seeded PRNG, SVG strings. The document is one string build — no script tags, no external references.</li>
</ul>
<div class="foot">attnlab v1.0.0 · generated deterministically · MIT © 2026 adithyanraj03</div>
</div></body></html>`);

  return b.join("\n");
}
