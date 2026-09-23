// SVG builders shared by the interactive app and the static report.
// Pure string functions of their inputs — the same weights always render
// the same bytes.

import { Mat, get } from "./mat.js";

const PAPER = "#f7f6f1";
const CELL_LO = [241, 240, 234]; // near-paper
const CELL_HI = [45, 40, 124]; // deep indigo
const MASK_FILL = "#fbfaf7";
const INK = "#252524";
const MUTED = "#676662";

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function heatColor(w: number): string {
  const t = Math.max(0, Math.min(1, w));
  const r = Math.round(lerp(CELL_LO[0], CELL_HI[0], t));
  const g = Math.round(lerp(CELL_LO[1], CELL_HI[1], t));
  const b = Math.round(lerp(CELL_LO[2], CELL_HI[2], t));
  return `rgb(${r},${g},${b})`;
}

export interface HeatmapOptions {
  cell: number; // px per cell
  causal: boolean; // gray out j > i
  labels: boolean; // token labels around the grid
  fontSize?: number;
  rowClick?: (i: number) => string; // if set, rows become clickable (returns the onclick attribute value)
  selectedRow?: number; // highlight one row
}

/**
 * An attention heatmap: rows = query positions, columns = key positions.
 * `tokens` supplies the labels (one per position).
 */
export function heatmapSVG(weights: Mat, tokens: string[], opts: HeatmapOptions): string {
  const { cell, causal, labels } = opts;
  const fs = opts.fontSize ?? Math.max(9, Math.round(cell * 0.3));
  const n = weights.r;
  const pad = labels ? cell * 0.9 + 4 : 2;
  const w = pad + n * cell + 2;
  const h = pad + n * cell + 2;
  let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" font-family="ui-monospace, Menlo, Consolas, monospace">`;
  s += `<rect x="0" y="0" width="${w}" height="${h}" fill="${PAPER}"/>`;
  if (labels) {
    for (let j = 0; j < n; j++) {
      s += `<text x="${pad + j * cell + cell / 2}" y="${pad - 6}" font-size="${fs}" fill="${MUTED}" text-anchor="middle">${esc(tokens[j])}</text>`;
    }
    for (let i = 0; i < n; i++) {
      s += `<text x="${pad - 6}" y="${pad + i * cell + cell / 2 + fs / 3}" font-size="${fs}" fill="${MUTED}" text-anchor="end">${esc(tokens[i])}</text>`;
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const x = pad + j * cell;
      const y = pad + i * cell;
      const masked = causal && j > i;
      const fill = masked ? MASK_FILL : heatColor(get(weights, i, j));
      s += `<rect x="${x + 0.5}" y="${y + 0.5}" width="${cell - 1}" height="${cell - 1}" fill="${fill}"/>`;
    }
  }
  if (opts.selectedRow !== undefined && opts.selectedRow >= 0 && opts.selectedRow < n) {
    const y = pad + opts.selectedRow * cell;
    s += `<rect x="${pad - 1}" y="${y - 1}" width="${n * cell + 2}" height="${cell + 2}" fill="none" stroke="${INK}" stroke-width="1.5"/>`;
  }
  if (opts.rowClick) {
    for (let i = 0; i < n; i++) {
      const y = pad + i * cell;
      s += `<rect x="${pad}" y="${y}" width="${n * cell}" height="${cell}" fill="transparent" onclick="${opts.rowClick(i)}"/>`;
    }
  }
  s += `</svg>`;
  return s;
}

export function barsSVG(values: number[], tokens: string[], color: string): string {
  const n = values.length;
  const rowH = 22;
  const labelW = 54;
  const maxW = 300;
  const w = labelW + maxW + 56;
  const h = n * rowH + 8;
  let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" font-family="ui-monospace, Menlo, Consolas, monospace">`;
  for (let j = 0; j < n; j++) {
    const y = 4 + j * rowH;
    const bw = Math.max(1, values[j] * maxW);
    s += `<text x="${labelW - 8}" y="${y + 13}" font-size="11" fill="${MUTED}" text-anchor="end">${esc(tokens[j])}</text>`;
    s += `<rect x="${labelW}" y="${y + 4}" width="${bw}" height="12" fill="${color}"/>`;
    s += `<text x="${labelW + bw + 6}" y="${y + 13}" font-size="11" fill="${INK}">${values[j].toFixed(3)}</text>`;
  }
  s += `</svg>`;
  return s;
}

export function esc(t: string): string {
  return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
