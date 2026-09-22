import { test } from "node:test";
import assert from "node:assert";
import { Rng } from "../dist/prng.js";

const KAT_SEED0 = [
  0xe220a8397b1dcdafn,
  0x6e789e6aa1b965f4n,
  0x06c45d188009454fn,
  0xf88bb8a8724c81ecn,
  0x1b39896a51a8749bn,
  0x53cb9f0c747ea2ean,
];
const KAT_STATE_AFTER_6 = 0xb54cda58fbbee87en;
const KAT_SEED7 = [
  0x63cbe1e459320dd7n,
  0x044c3cd7f43c661cn,
  0xe6984080bab12a02n,
  0x953aeb70673e29cbn,
];

test("KAT: seed 0 first six u64 outputs match the VecLab (Rust) and Strainlab (Go) reference values", () => {
  const r = new Rng(0);
  for (let i = 0; i < 6; i++) {
    assert.strictEqual(r.nextU64(), KAT_SEED0[i], `draw ${i + 1}`);
  }
});

test("KAT: state after six draws from seed 0", () => {
  const r = new Rng(0);
  for (let i = 0; i < 6; i++) r.nextU64();
  assert.strictEqual(r.state(), KAT_STATE_AFTER_6);
});

test("KAT: seed 7 first four u64 outputs", () => {
  const r = new Rng(7);
  for (let i = 0; i < 4; i++) {
    assert.strictEqual(r.nextU64(), KAT_SEED7[i], `draw ${i + 1}`);
  }
});

test("two streams with the same seed are bit-identical", () => {
  const a = new Rng(1234);
  const b = new Rng(1234);
  for (let i = 0; i < 1000; i++) {
    assert.strictEqual(a.nextU64(), b.nextU64());
  }
});

test("different seeds diverge", () => {
  const a = new Rng(1);
  const b = new Rng(2);
  let same = 0;
  for (let i = 0; i < 64; i++) if (a.nextU64() === b.nextU64()) same++;
  assert.ok(same <= 1, `seeds 1 and 2 matched ${same}/64 draws`);
});

test("nextFloat is in [0, 1) over 200k draws", () => {
  const r = new Rng(9);
  for (let i = 0; i < 200000; i++) {
    const v = r.nextFloat();
    assert.ok(v >= 0 && v < 1, `draw ${i} out of range: ${v}`);
  }
});

test("nextFloat mean is near 0.5", () => {
  const r = new Rng(11);
  let sum = 0;
  const n = 100000;
  for (let i = 0; i < n; i++) sum += r.nextFloat();
  assert.ok(Math.abs(sum / n - 0.5) < 0.005);
});

test("nextInt stays in [0, n)", () => {
  const r = new Rng(5);
  for (let i = 0; i < 100000; i++) {
    const v = r.nextInt(97);
    assert.ok(v >= 0 && v < 97, `draw ${i} out of range: ${v}`);
  }
});

test("nextInt covers all residues", () => {
  const r = new Rng(3);
  const seen = new Array(7).fill(false);
  for (let i = 0; i < 7000; i++) seen[r.nextInt(7)] = true;
  assert.ok(seen.every(Boolean), "some residue never drawn");
});

test("nextInt rejects non-positive n", () => {
  assert.throws(() => new Rng(1).nextInt(0));
  assert.throws(() => new Rng(1).nextInt(2.5));
});

test("nextGaussian has mean ~0 and std ~1 over 100k samples", () => {
  const r = new Rng(17);
  let sum = 0;
  let sumsq = 0;
  const n = 100000;
  for (let i = 0; i < n; i++) {
    const v = r.nextGaussian();
    assert.ok(Number.isFinite(v), `draw ${i} not finite: ${v}`);
    sum += v;
    sumsq += v * v;
  }
  const mean = sum / n;
  const std = Math.sqrt(sumsq / n - mean * mean);
  assert.ok(Math.abs(mean) < 0.02, `mean ${mean}`);
  assert.ok(std > 0.95 && std < 1.05, `std ${std}`);
});

test("nextSpan is in [-1, 1)", () => {
  const r = new Rng(19);
  for (let i = 0; i < 100000; i++) {
    const v = r.nextSpan();
    assert.ok(v >= -1 && v < 1, `draw ${i} out of range: ${v}`);
  }
});
