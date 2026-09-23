// The interactive playground. Builds its own DOM, recomputes the full
// forward pass on every control change, and renders SVG heatmaps. No
// Math.random anywhere: even the "reseed" button steps a fixed LCG, so a
// click sequence is reproducible.

import { DEFAULT_CONFIG, ForwardResult, PlaygroundConfig, SCENARIOS, forward, tokenColor } from "./model.js";
import { heatmapSVG, barsSVG } from "./svg.js";
import { get } from "./mat.js";

const VERSION = "1.0.0";

interface AppState {
  cfg: PlaygroundConfig;
  row: number; // selected query row for the trace panel
}

const state: AppState = { cfg: { ...DEFAULT_CONFIG }, row: 0 };
let last: ForwardResult | null = null;

function lcgStep(seed: number): number {
  return (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff;
}

function css(): string {
  return `
  body{margin:0;background:#f7f6f1;color:#252524;font-family:Georgia,'Times New Roman',serif;}
  .al{max-width:1180px;margin:0 auto;padding:36px 36px 26px;}
  .al h1{font-size:27px;margin:0 0 5px;font-weight:600;}
  .al .meta{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10.5px;letter-spacing:.12em;color:#676662;text-transform:uppercase;margin-bottom:24px;}
  .al .cols{display:grid;grid-template-columns:252px 1fr;gap:22px;align-items:start;}
  .panel{background:#fff;border:1px solid #dcdad1;border-radius:10px;padding:16px;box-shadow:0 1px 2px rgba(37,37,36,.06);}
  .panel h2{font-size:14px;margin:0 0 12px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:#676662;}
  .ctl{margin-bottom:13px;}
  .ctl label{display:block;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;color:#676662;margin-bottom:4px;text-transform:uppercase;letter-spacing:.05em;}
  .ctl select,.ctl input[type=number]{width:100%;box-sizing:border-box;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;padding:5px 7px;border:1px solid #dcdad1;border-radius:7px;background:#fff;color:#252524;}
  .ctl input[type=range]{width:100%;accent-color:#5b51c7;}
  .ctl .val{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;color:#252524;}
  .chk{display:flex;align-items:center;gap:8px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;}
  .chk input{accent-color:#5b51c7;}
  .btn{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;padding:6px 12px;border:1px solid #5b51c7;background:#5b51c7;color:#fff;border-radius:7px;cursor:pointer;}
  .btn:hover{background:#4a41b0;}
  .tokens{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:16px;}
  .tok{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;padding:4px 10px;border-radius:999px;color:#fff;cursor:pointer;border:2px solid transparent;}
  .tok.sel{border-color:#252524;}
  .heads{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px;}
  .head{background:#fff;border:1px solid #dcdad1;border-radius:10px;padding:14px;box-shadow:0 1px 2px rgba(37,37,36,.06);}
  .head h3{margin:0 0 2px;font-size:14.5px;font-weight:600;}
  .head .stat{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;color:#676662;margin-bottom:8px;}
  .head svg{display:block;max-width:100%;height:auto;}
  .trace .stat{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;color:#676662;margin:0 0 8px;}
  .foot{margin-top:26px;padding-top:13px;border-top:1px solid #dcdad1;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10.5px;color:#676662;}
  `;
}

function controlHTML(cfg: PlaygroundConfig): string {
  const sel = (v: boolean) => (v ? " selected" : "");
  return `
  <div class="ctl"><label>Scenario</label><select id="al-scenario">
    ${SCENARIOS.map((s) => `<option value="${s.name}"${sel(s.name === cfg.scenario)}>${s.name}</option>`).join("")}
  </select></div>
  <div class="ctl"><label>Preset</label><select id="al-preset">
    <option value="structured"${sel(cfg.preset === "structured")}>structured (designed heads)</option>
    <option value="random"${sel(cfg.preset === "random")}>random (seeded baseline)</option>
  </select></div>
  <div class="ctl"><label>Heads — <span class="val" id="al-heads-v">${cfg.heads}</span></label>
    <input type="range" id="al-heads" min="1" max="4" step="1" value="${cfg.heads}"></div>
  <div class="ctl"><label>Head dim — <span class="val" id="al-hd-v">${cfg.headDim}</span></label>
    <input type="range" id="al-hd" min="4" max="16" step="1" value="${cfg.headDim}"></div>
  <div class="ctl"><label>Temperature — <span class="val" id="al-t-v">${cfg.temperature.toFixed(2)}</span></label>
    <input type="range" id="al-t" min="0.25" max="4" step="0.25" value="${cfg.temperature}"></div>
  <div class="ctl"><div class="chk"><input type="checkbox" id="al-causal"${cfg.causal ? " checked" : ""}><label for="al-causal" style="margin:0;">causal mask</label></div></div>
  <div class="ctl"><label>Seed</label>
    <div style="display:flex;gap:7px;"><input type="number" id="al-seed" value="${cfg.seed}">
    <button class="btn" id="al-dice" title="deterministic LCG step — never Math.random" style="white-space:nowrap;">step</button></div></div>
  <div style="font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10.5px;color:#9a9994;line-height:1.6;margin-top:6px;">
  every control change recomputes the full forward pass. same config + seed ⇒ identical matrices.</div>`;
}

function render(r: ReturnType<typeof forward>, row: number): string {
  const heads = r.plans
    .map((p, h) => {
      const w = r.multi.perHead[h].weights;
      const pk = r.peaks[h];
      const svg = heatmapSVG(w, r.tokens, {
        cell: 34,
        causal: r.cfg.causal,
        labels: true,
        fontSize: 10,
        selectedRow: row,
        rowClick: (i) => `Attnlab.selectRow(${i})`,
      });
      return `<div class="head">
        <h3>head ${h} · ${p.name}${p.kind === "random" ? " (seeded)" : ""}</h3>
        <div class="stat">entropy ${r.entropies[h].toFixed(2)} nats · peak ${pk.value.toFixed(3)} @ ${pk.row}→${pk.col}</div>
        ${svg}
      </div>`;
    })
    .join("\n");

  const traceVals: number[] = [];
  for (let j = 0; j < r.n; j++) traceVals.push(get(r.multi.perHead[0].weights, row, j));

  return `
  <div class="tokens">
    ${r.tokens
      .map((t, i) => `<span class="tok${i === row ? " sel" : ""}" style="background:${tokenColor(r.typeOf[i])}" onclick="Attnlab.selectRow(${i})">${i + 1} · ${t}</span>`)
      .join("")}
  </div>
  <div class="heads">${heads}</div>
  <div class="head trace" style="margin-top:16px;">
    <h3>query ${row} · “${r.tokens[row]}” — head 0 attention over keys</h3>
    <div class="stat">click any token or heatmap row to re-point the query</div>
    ${barsSVG(traceVals, r.tokens, "#5b51c7")}
  </div>`;
}

function refresh(): void {
  if (typeof document === "undefined") return; // no DOM (node tests)
  const r = forward(state.cfg);
  last = r;
  const stage = document.getElementById("al-stage");
  const ctrls = document.getElementById("al-ctrl");
  if (!stage || !ctrls) return;
  stage.innerHTML = render(r, Math.min(state.row, r.n - 1));
  ctrls.innerHTML = controlHTML(state.cfg);
  wireControls();
}

function wireControls(): void {
  const on = (id: string, fn: (el: HTMLElement) => void) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("input", () => fn(el));
  };
  on("al-scenario", (el) => { state.cfg.scenario = (el as HTMLSelectElement).value; refresh(); });
  on("al-preset", (el) => { state.cfg.preset = (el as HTMLSelectElement).value as PlaygroundConfig["preset"]; refresh(); });
  on("al-heads", (el) => { state.cfg.heads = Number((el as HTMLInputElement).value); refresh(); });
  on("al-hd", (el) => { state.cfg.headDim = Number((el as HTMLInputElement).value); refresh(); });
  on("al-t", (el) => { state.cfg.temperature = Number((el as HTMLInputElement).value); refresh(); });
  on("al-causal", (el) => { state.cfg.causal = (el as HTMLInputElement).checked; refresh(); });
  on("al-seed", (el) => { state.cfg.seed = Number((el as HTMLInputElement).value); refresh(); });
  const dice = document.getElementById("al-dice");
  if (dice) dice.addEventListener("click", () => { state.cfg.seed = lcgStep(state.cfg.seed); refresh(); });
}

function init(): void {
  if (typeof document === "undefined") return; // no DOM (node tests)
  const app = document.getElementById("al-root");
  if (!app) return;
  const st = document.createElement("style");
  st.textContent = css();
  document.head.appendChild(st);
  app.innerHTML = `
  <div class="al">
    <h1>Attnlab</h1>
    <div class="meta">real multi-head attention · computed live in your browser · hand-rolled matrix code · zero dependencies</div>
    <div class="cols">
      <div class="panel" id="al-ctrl"></div>
      <div id="al-stage"></div>
    </div>
    <div class="foot">attnlab v${VERSION} · softmax(QKᵀ/(√d·T))·V · deterministic — same config + seed ⇒ identical matrices · MIT © 2026 adithyanraj03</div>
  </div>`;
  refresh();
}

export const Attnlab = {
  version: VERSION,
  state,
  init,
  refresh,
  selectRow(i: number): void {
    state.row = i;
    if (last && typeof document !== "undefined") {
      const stage = document.getElementById("al-stage");
      if (stage) stage.innerHTML = render(last, i);
    }
  },
  _last: () => last,
  _lcgStep: lcgStep,
};

// expose on window for inline onclick handlers
declare global {
  interface Window {
    Attnlab: typeof Attnlab;
  }
}
if (typeof window !== "undefined") {
  (window as Window).Attnlab = Attnlab;
}
