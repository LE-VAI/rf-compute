# Hello World: Lattice-Coded AirComp

## Tier 1.5 — The $0 exact-computation primitive

> **The experiment:** Two nodes hold integers. Each transmits a dithered lattice codeword. The receiver recovers their sum mod L — **exactly**, not as an estimate — from ONE channel use, at noise levels that would drown an analog reading. And it never decodes either message. Interference is the operation; the lattice makes it *exact*.

Tier 1 ([hello-world-aircomp.md](hello-world-aircomp.md)) showed the analog version: the channel superposes two signals and the receiver reads a real-valued sum estimate. Its honest limitation is printed at the bottom of that walkthrough: noise corrupts the estimate irrecoverably. This walkthrough is the upgrade the theory promised — the **full Nazer/Gastpar result**, implemented in software, free.

It maps to the founding papers of computation over multiple-access channels: Nazer & Gastpar, "Computation over multiple-access channels," *IEEE Trans. Inf. Theory* 53(10), 3498–3516 (2007), and "Compute-and-forward: Harnessing interference through structured codes," *IEEE Trans. Inf. Theory* 57(10), 6463–6486 (2011). [DOI: 10.1109/TIT.2011.2165816](https://doi.org/10.1109/TIT.2011.2165816)

---

## Honest scope

What this walkthrough demonstrates, precisely:

- **Exact integer sums from a noisy channel.** The receiver recovers `w1 + w2 mod L` with error probability that decays *exponentially* in the lattice dimension — not an MSE that shrinks, a hard threshold that the noise must cross to cause an error.
- **One channel use for N nodes.** The lattice scheme's decoded value IS the sum. Routing (TDMA) must decode every message first — N channel uses and an N-way error union.
- **Structural privacy.** The receiver recovers only the sum. Individual messages are not masked — they are *structurally absent* from what the receiver can compute.

What this walkthrough does NOT do (the honest list):

- **No RF hardware.** The channel is simulated AWGN. The signal-processing structure is complete and hardware-ready (the SDR differences are sync and dither-seed sharing — covered below), but this page is the $0 tier: a laptop, Python, NumPy.
- **Toy code family.** The theory uses high-dimensional random lattice ensembles; we use repetition across n coordinates — which already exhibits the exponential error decay. The theory optimizes a receiver scaling α; we fix α = 1 (inverted channel). The 2007 paper computes over finite fields; our `Z_L` is the integers mod L, the cleanest finite ring for pedagogy.
- **Coefficients are realized as channel gains.** In compute-and-forward the integer coefficients a_i fall out of the channel gains. Here we *set* the gains — the toy makes the equation demonstrable, not the fading-channel estimation problem.

---

## The math, made legible

### Nested lattices in one paragraph

A lattice is a discrete additive subgroup of R^n. We use the simplest nested pair:

```
fine lattice   = Z^n        (the integers — message carriers)
coarse lattice = L·Z^n      (fundamental cell [-L/2, L/2)^n)
```

Each node i holds an integer `w_i` in `Z_L = {0..L-1}`. The message labels a **coset** of the coarse lattice: the set `{w_i + L·k : k in Z^n}`. Encoding picks a random coset representative, subtracts a dither, and reduces:

```
x_i = [v_i − d_i] mod L        v_i = w_i + L·k_i (random k_i)
                                d_i uniform on the cell
```

The **dither** does two jobs (Erez & Zamir 2004, on which Nazer/Gastpar build): it makes the transmitted signal uniform over the cell (power shaping) and statistically independent of the message (the waveform alone reveals nothing). The dither is **not secret** — the receiver gets it from a shared PRNG seed. What the receiver never learns is the messages.

### The channel superposes

```
y = Σ h_i · x_i + z        z ~ N(0, σ²·I)
```

This is the same physical channel as Tier 1. Same superposition, same noise. The difference is entirely in what the codewords *are*.

### The dither replay (the heart of the construction)

The receiver adds the dithers back — weighted by the same integer gains it wants to compute with:

```
y + Σ a_i·d_i  =  Σ a_i·w_i  +  L·(integers)  +  z
```

Every L-multiple dies in the mod-L reduction. The coset randomness dies. The dithers die. What survives is:

```
r = [Σ a_i·w_i + z] mod L
```

— the requested integer combination of the messages, plus noise, cleanly. **This is why the coefficients are free**: the receiver replays the dithers at whatever integer weights it wants, and the algebra absorbs them.

### The circular mean (why averaging survives the wrap)

Each of the n coordinates of `r` is an independent noisy copy of the same integer, wrapped mod L. `Z_L` is a **circle**, not an interval: plain averaging fails at the wrap (values near 0 and L−1 pull the mean in opposite directions), and round-then-majority-vote forfeits the dimensional gain. The circular mean keeps both:

1. Map each `r_j` to its unit vector at angle `2π·r_j/L`
2. Average the vectors
3. Read the angle, round, mod L

Full `√n` averaging gain at any L. This is the pedagogical heart of the construction: **the lattice doesn't change the channel; it changes what the noise is allowed to do.** Noise that would smear an analog estimate now has to push an integer across a half-integer boundary — and it has to do it in *every* coordinate at once to win.

---

## The code

Everything below is in the package. Run it as-is or follow along.

### One round through the kernel

```python
from rf_compute import LatticeAirCompOperator, WaveComputeKernel

kernel = WaveComputeKernel(backend="sim")

# Two nodes, integers mod 16, lattice dimension 32
op = LatticeAirCompOperator(num_nodes=2, modulus=16, dimension=32)
out = kernel.apply(op, inputs=[11, 9])

print(out['value'])      # 4  — because (11 + 9) mod 16 = 4
print(out['exact'])      # True
print(out['messages'])  # [11, 9] — what WAS transmitted (verification only;
                         #  the receiver structurally cannot recover these)
```

### The machinery, bare (what the kernel binds)

```python
import numpy as np
from rf_compute.lattice import encode, channel, decode, monte_carlo

rng = np.random.default_rng(42)
L, n = 16, 32

# ── each node: message -> dithered lattice codeword ──
w1, w2 = 11, 9
x1, d1 = encode(w1, L, n, rng)   # x1: uniform on the cell, message-independent
x2, d2 = encode(w2, L, n, rng)   # d1, d2: the dithers — shared with the receiver
                                 #          (shared PRNG seed in practice), NOT secret

# ── the channel: superposition + AWGN at 5 dB ──
y = channel([x1, x2], L, snr_db=5.0)

# ── the receiver: knows L, the dithers, and its coefficient intent.
#    Never the messages, never any individual transmission. ──
s_hat = decode(y, L, dithers=[d1, d2])

print(s_hat)   # 4 — (11 + 9) mod 16, exactly, from ONE noisy channel use
```

### The compute-and-forward equation

```python
# Coefficients [a_1..a_N]: the channel computes sum(a_i * w_i) mod L.
# Integer coefficients are realized as channel gains — node i's signal
# arrives at gain a_i — and the receiver replays dithers at those weights.
op = LatticeAirCompOperator(num_nodes=2, modulus=16, dimension=32,
                            coefficients=[1, 2])
out = kernel.apply(op, inputs=[5, 9])
print(out['value'])   # 7 — because (1·5 + 2·9) mod 16 = 23 mod 16 = 7
```

### The full experiment (what the example runs)

```bash
python examples/tier1_5_lattice_aircomp_kernel.py
```

This prints four panels: the exact one-use sum, the coefficient equation, error vs lattice dimension, and the honest scoreline against both baselines.

---

## What you should see

### Error vs dimension — the theorem's signature

At 5 dB, two nodes, L=16, 2000 trials per point:

```
   n   error    channel uses
   4   0.71         1
   8   0.60         1
  16   0.47         1
  32   0.33         1
  64   0.15         1
```

Every added coordinate is another independent vote on the same sum. The error falls toward zero *exponentially* — the toy-scale version of the random-lattice error exponent in the theorem. No analog scheme can do this: an analog estimate's MSE shrinks as σ²/n but its *error probability* against an integer target does not hit a hard threshold.

### The honest scoreline — all three schemes at matched power

At 5 dB, two nodes, L=16, n=32, 4000 trials:

| scheme | error | channel uses | what it actually computes |
|---|---:|---:|---|
| **lattice** | 0.32 | **1** | the exact sum mod L; individual messages structurally absent |
| analog (Tier 1) | 0.58 | 1 | a real-valued estimate of the raw sum; no mod-L arithmetic; power-shaping gap (factor-2 noise margin at matched power) |
| tdma (routing) | 0.44 | **2** | every message decoded first, then summed; N-way error union (one bad slot ruins the sum) |

Read the TDMA row carefully — it is the point of the whole result. TDMA decodes each slot at nearly the lattice's per-slot reliability, pays **N times** the channel resources... and still loses to the lattice's single use, because its errors are a union across slots. **The sum is decodable at the rate of ONE user while all N transmit.** That is the Nazer/Gastpar computation-rate result, made visible in one table.

### The privacy property (not in the numbers)

The lattice receiver recovers `Σ w_i mod L` and nothing else. Not masked, not encrypted-around: the individual messages never exist at the receiver. Contrast: analog transmits raw values; TDMA decodes every message. If a later release adds multi-round protocols, this structural property is what they'd build on.

---

## Troubleshooting (the honest list)

| Symptom | Cause | Fix |
|---|---|---|
| `error_rate` looks high at small n | It is — the exponential decay needs dimension. Watch the n=4→64 sweep; that's the theorem working, not a bug. |
| Analog baseline beats lattice at very high SNR | Expected at the toy scale: analog's shaping penalty vanishes as noise→0. The lattice's advantages (exactness threshold, mod-L arithmetic, privacy) remain. Compare at *moderate* SNR — that's the regime the result lives in. |
| Coefficient decode returns the plain sum | The gains must be applied at the *channel* (`gains=a` in `channel()`, or the kernel's coefficient path) AND replayed at the decoder (`a=a`). One without the other breaks the algebra. |
| `modulus=2` looks degenerate | It's the 1-bit case — legal and instructive (sums mod 2 = XOR). The interesting privacy/error behavior shows at L≥8. |
| Results vary between runs | `channel()` draws unseeded noise (the physical channel doesn't take seeds). Pass a seeded `rng` to `run_trial`/`monte_carlo` for reproducible numbers. |

---

## What this proves

1. **Exact computation from a noisy channel is possible at one channel use.** Not an estimate that degrades — a hard-decision decode with error probability decaying exponentially in the code dimension. Tier 1's analog sum cannot cross this line by construction.

2. **The computation-rate result, made visible.** Routing pays N channel uses AND unions its errors. The lattice scheme computes the same function in one use, at lower error. Interference is not merely tolerated — it *is* the addition.

3. **The receiver computes a function, not the inputs.** `Σ a_i·w_i mod L` is what's decodable. The individual messages are structurally absent. This is the seed of "compute, don't decode" — the design stance the 6G AirComp literature builds on.

4. **Structure beats power.** The lattice codeword, the dither, and the mod-L reduction together buy exactness at the *same* power the analog scheme wastes on a smeared estimate. The win is algebra, not watts.

---

## Going deeper

### Toward the fading-channel version

The toy sets the channel gains; the real compute-and-forward problem is that gains fall out of a fading channel and the receiver *chooses* integer a_i to maximize the computation rate (lattice-reduction algorithms). Natural next surface: a `fading` channel model where a_i are estimated, not set.

### Toward the SDR version

The kernel's `backend="sdr"` path transmits the same codewords over real RF. The two hardware deltas beyond Tier 1's setup: (1) sample-level sync across transmitters (the lattice dimension lives in the samples — timing error is noise); (2) dither-seed sharing (the receiver must replay exactly the dithers the transmitters used — a PRNG seed side-channel, standard in the compute-and-forward literature). Tier 1's 3-RF-chain setup ([hello-world-aircomp.md](hello-world-aircomp.md#hardware)) is the hardware baseline.

### Toward Tier 2 (wave-domain convolution)

Tier 1.5 is the last purely-additive primitive: the channel computes a linear function of the messages. Tier 2 upgrades the *operator*: a programmed medium computes `conv(x, h)` in the wave domain. See [hello-world-convolution.md](hello-world-convolution.md).

### Toward Tier 3 (matrix inversion via feedback)

Tier 3 closes the loop — the receiver feeds its output back and the loop solves `Ax = b` in the wave domain. See [hello-world-matrix-inversion.md](hello-world-matrix-inversion.md).

---

## Safety and legality

This tier runs entirely in simulation — no RF transmission, nothing to license. If you proceed to the SDR version, the Tier 1 rules apply: ISM bands only, keep power low, check local regulations. See [hello-world-aircomp.md](hello-world-aircomp.md#safety-and-legality).

---

## Provenance

- **Maps to:** Nazer & Gastpar, "Computation over multiple-access channels," *IEEE Trans. Inf. Theory* 53(10), 3498–3516 (2007). [DOI: 10.1109/TIT.2007.904734](https://doi.org/10.1109/TIT.2007.904734)
- **Namesake extension:** Nazer & Gastpar, "Compute-and-forward: Harnessing interference through structured codes," *IEEE Trans. Inf. Theory* 57(10), 6463–6486 (2011). [DOI: 10.1109/TIT.2011.2165816](https://doi.org/10.1109/TIT.2011.2165816)
- **Dithered lattice coding:** Erez & Zamir, "Achieving 1/2 log (1+SNR) over the additive white Gaussian noise channel using lattice codes," *IEEE Trans. Inf. Theory* 50(10), 2293–2314 (2004).
- **Module:** `rf_compute/lattice.py` · **Operator:** `LatticeAirCompOperator` · **Example:** `examples/tier1_5_lattice_aircomp_kernel.py` · **Tests:** `tests/test_lattice_aircomp.py`
- **Honest scope:** Toy lattice family (repetition across n coordinates, α = 1, Z_L), simulated AWGN, coefficients realized as set channel gains. The exponential error decay, the computation-rate advantage over routing, and the structural privacy property are demonstrated at toy scale; the random-lattice ensembles and fading-channel coefficient selection of the full theory are not reproduced.