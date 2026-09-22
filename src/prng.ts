// SplitMix64 — the same stream family the VecLab (Rust) and Strainlab (Go)
// projects use, so the whole lab behaves like one instrument. JavaScript
// numbers are 53-bit, so the 64-bit mixing runs on BigInt; everything else
// consumes plain numbers.

const SM_INC = 0x9e3779b97f4a7c15n;
const SM_M1 = 0xbf58476d1ce4e5b9n;
const SM_M2 = 0x94d049bb133111ebn;
const U64_MAX = (1n << 64n) - 1n;

export class Rng {
  private s: bigint;

  constructor(seed: number | bigint) {
    this.s = BigInt(seed) & U64_MAX;
  }

  /** Current state — fully determines the remaining stream. */
  state(): bigint {
    return this.s;
  }

  /** Next full 64-bit SplitMix64 output. */
  nextU64(): bigint {
    this.s = (this.s + SM_INC) & U64_MAX;
    let z = this.s;
    z = ((z ^ (z >> 30n)) * SM_M1) & U64_MAX;
    z = ((z ^ (z >> 27n)) * SM_M2) & U64_MAX;
    return (z ^ (z >> 31n)) & U64_MAX;
  }

  /** Uniform float in [0, 1) with 53 bits of resolution. */
  nextFloat(): number {
    return Number(this.nextU64() >> 11n) / 2 ** 53;
  }

  /** Uniform integer in [0, n). Throws for n <= 0.
   * Rejection sampling on the low 32 bits keeps bias under 1e-9. */
  nextInt(n: number): number {
    if (!Number.isInteger(n) || n <= 0) throw new Error("Rng.nextInt: n must be a positive integer");
    const mod = 2 ** 32;
    const limit = mod - (mod % n);
    for (;;) {
      const x = Number(this.nextU64() & 0xffffffffn);
      if (x < limit) return x % n;
    }
  }

  /** Standard normal via Box-Muller (cosine branch), same convention as
   * the Rust and Go ports. */
  nextGaussian(): number {
    const u = this.nextFloat();
    const v = this.nextFloat();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** Uniform in [-1, 1). */
  nextSpan(): number {
    return this.nextFloat() * 2 - 1;
  }
}
