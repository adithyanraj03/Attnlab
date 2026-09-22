// Hand-rolled matrix math. Row-major Float64Array storage, no dependency.
// Every operation is a pure function of its inputs.

export interface Mat {
  r: number; // rows
  c: number; // columns
  a: Float64Array; // length r*c, row-major
}

export function mat(r: number, c: number): Mat {
  if (!Number.isInteger(r) || !Number.isInteger(c) || r < 0 || c < 0) {
    throw new Error(`mat: invalid dimensions ${r}x${c}`);
  }
  return { r, c, a: new Float64Array(r * c) };
}

export function fromRows(rows: number[][]): Mat {
  const r = rows.length;
  const c = rows[0]?.length ?? 0;
  const m = mat(r, c);
  for (let i = 0; i < r; i++) {
    if (rows[i].length !== c) throw new Error(`fromRows: ragged row ${i}`);
    for (let j = 0; j < c; j++) m.a[i * c + j] = rows[i][j];
  }
  return m;
}

export function matmul(x: Mat, y: Mat): Mat {
  if (x.c !== y.r) throw new Error(`matmul: ${x.r}x${x.c} * ${y.r}x${y.c}`);
  const out = mat(x.r, y.c);
  for (let i = 0; i < x.r; i++) {
    const xo = i * x.c;
    const oo = i * y.c;
    for (let k = 0; k < x.c; k++) {
      const xik = x.a[xo + k];
      if (xik === 0) continue;
      const yo = k * y.c;
      for (let j = 0; j < y.c; j++) out.a[oo + j] += xik * y.a[yo + j];
    }
  }
  return out;
}

export function addm(x: Mat, y: Mat): Mat {
  if (x.r !== y.r || x.c !== y.c) throw new Error("addm: dimension mismatch");
  const out = mat(x.r, x.c);
  for (let i = 0; i < x.a.length; i++) out.a[i] = x.a[i] + y.a[i];
  return out;
}

export function scalem(x: Mat, s: number): Mat {
  const out = mat(x.r, x.c);
  for (let i = 0; i < x.a.length; i++) out.a[i] = x.a[i] * s;
  return out;
}

export function transpose(x: Mat): Mat {
  const out = mat(x.c, x.r);
  for (let i = 0; i < x.r; i++) {
    for (let j = 0; j < x.c; j++) out.a[j * x.r + i] = x.a[i * x.c + j];
  }
  return out;
}

/** Copy of row i (a fresh Float64Array; mutation never touches the source). */
export function row(x: Mat, i: number): Float64Array {
  if (i < 0 || i >= x.r) throw new Error(`row: index ${i} out of range`);
  return x.a.slice(i * x.c, (i + 1) * x.c);
}

/** x[i, j] */
export function get(x: Mat, i: number, j: number): number {
  return x.a[i * x.c + j];
}

/**
 * Stable softmax of one row of logits, scaled by 1/temperature.
 * Temperature must be > 0. As temperature grows the result approaches
 * uniform; as it shrinks it approaches the argmax one-hot.
 */
export function softmax(logits: Float64Array, temperature: number): Float64Array {
  if (temperature <= 0) throw new Error("softmax: temperature must be > 0");
  const out = new Float64Array(logits.length);
  let max = -Infinity;
  for (let j = 0; j < logits.length; j++) {
    if (logits[j] > max) max = logits[j];
  }
  let sum = 0;
  for (let j = 0; j < logits.length; j++) {
    const e = Math.exp((logits[j] - max) / temperature);
    out[j] = e;
    sum += e;
  }
  if (sum === 0 || !Number.isFinite(sum)) throw new Error("softmax: no finite mass (all -Infinity?)");
  for (let j = 0; j < logits.length; j++) out[j] /= sum;
  return out;
}

/** True where j >= i (the allowed "past" region for a causal row i). */
export function causalMask(n: number): boolean[][] {
  const m: boolean[][] = [];
  for (let i = 0; i < n; i++) {
    const rowm: boolean[] = [];
    for (let j = 0; j < n; j++) rowm.push(j <= i);
    m.push(rowm);
  }
  return m;
}

/** First index of the maximum (ties go to the lowest index — deterministic). */
export function argmax(v: Float64Array): number {
  let bi = 0;
  for (let j = 1; j < v.length; j++) if (v[j] > v[bi]) bi = j;
  return bi;
}

/** Shannon entropy in nats; 0·log(0) is 0 by convention. */
export function entropy(v: Float64Array): number {
  let h = 0;
  for (let j = 0; j < v.length; j++) {
    const p = v[j];
    if (p > 0) h -= p * Math.log(p);
  }
  return h;
}
