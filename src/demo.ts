// The terminal demo: prints the type-head attention matrix of the standard
// state plus one summary line per head. Pure function of the config.

import { DEFAULT_CONFIG, PlaygroundConfig, forward, fmtW } from "./model.js";
import { get } from "./mat.js";

export function runDemo(cfg: PlaygroundConfig = DEFAULT_CONFIG): string {
  const r = forward(cfg);
  const L: string[] = [];
  const sep = "─".repeat(78);

  L.push(`attnlab v1.0.0 · attention playground · seed ${r.cfg.seed} · deterministic`);
  L.push(sep);
  L.push(
    `${r.cfg.scenario} · ${r.n} tokens · ${r.vocab.length} types · ${r.heads} heads × d ${r.headDim} · causal ${r.cfg.causal ? "on" : "off"} · T ${r.cfg.temperature.toFixed(2)}`,
  );
  L.push(`tokens: ${r.tokens.join(" ")}`);
  L.push("");

  // find the type head (first "type" plan), print its full matrix
  const typeIdx = r.plans.findIndex((p) => p.kind === "type");
  const shown = typeIdx >= 0 ? typeIdx : 0;
  const w = r.multi.perHead[shown].weights;
  L.push(`${r.plans[shown].name} head — attention weights by key position (rows = queries)`);
  const colHdr = "      " + Array.from({ length: r.n }, (_, j) => String(j).padStart(3)).join(" ");
  L.push(colHdr);
  for (let i = 0; i < r.n; i++) {
    const cells: string[] = [];
    for (let j = 0; j < r.n; j++) cells.push(fmtW(get(w, i, j)).padStart(4));
    L.push(`${String(i).padStart(3)}  ` + cells.join(" "));
  }

  // summary lines for the other heads
  for (let h = 0; h < r.plans.length; h++) {
    if (h === shown) continue;
    const pk = r.peaks[h];
    L.push(
      `${r.plans[h].name.padEnd(8)} — entropy ${r.entropies[h].toFixed(2)} nats · peak ${fmtW(pk.value)} (query ${pk.row} → key ${pk.col})`,
    );
  }

  L.push(sep);
  L.push("deterministic: same seed → identical matrix (verified by the test suite)");
  return L.join("\n");
}
