# Hello World: Fading-Channel Coefficients

## Tier 1.6 — The coefficients must be earned

> **The experiment:** Three nodes transmit over a fading channel. The receiver cannot decode their sum — the gains don't line up. So it *chooses* an integer combination it CAN decode: it searches for the coefficient vector maximizing the computation rate, finds one, pre-codes it, and the channel computes that combination instead. On 94.6% of channels the plain sum has rate ZERO; the selected equation is decodable.

Tier 1.5 ([hello-world-lattice-aircomp.md](hello-world-lattice-aircomp.md)) made the compute-and-forward equation exact — by *setting* the channel gains to integers: `h_i = a_i`. That is the theory's cleanest case, and it hides the problem the coefficient vector exists to solve. On a real channel the gains fade (`h_i ~ N(0,1)`), and no integer vector is handed to you: you must find one.

This is the tier where the algorithm lives. It maps to the compute-and-forward coefficient-selection lineage: Nazer & Gastpar's founding theorem and its norm bound (*IEEE Trans. Inf. Theory* 57(10), 6463–6486, 2011), Sahraei & Gastpar's exact polynomial algorithm (Allerton 2014, [arXiv:1410.3656](https://arxiv.org/abs/1410.3656)), Liu & Ling's efficient integer search for complex channels (*IEEE Trans. Wireless Commun.* 15(12), 8039–8050, 2016), and the MMSE scaling and rate formula of Huang & Burr ([arXiv:1704.05007](https://arxiv.org/abs/1704.05007)).

---

## Honest scope

What this walkthrough demonstrates, precisely:

- **Coefficient selection as an optimization.** The computation rate `R(h, a)` is computable in closed form for any `a`; the receiver maximizes it over integer vectors. Three methods ship: exhaustive (the theorem's own search), LLL-aided (polynomial-time, the practical choice), and nearest-integer (the baseline).
- **The plain sum is usually not decodable.** On a fading channel the `a = 1` vector has rate 0 on ~95% of realizations — its alignment is wrong for the realized channel. Selection finds a decodable equation where there is one.
- **Selection saves power.** Inverting a deep fade costs `1/h_i²`; selection can drop the node (`a_i = 0`) instead. Median power cost: 0.39× baseline for selection vs 6.27× for the plain sum.

What this walkthrough does NOT do (the honest list):

- **No hardware.** Simulated fading; the signal-processing structure is complete, the RF part is Tier 1's setup.
- **The toy inverts the channel.** Node `i` pre-codes `(a_i/h_i)·x_i`, so the effective gain is exactly the integer `a_i` and the scalar-repetition decoder stays exact. On the real (non-inverted) channel the receiver would scale by real `α` and decode in R^n with a genuine lattice decoder; that decoder is what the rate formula describes, and it is **not implemented here** — the rate numbers are the theory's, the decode path is the inverted toy's.
- **Real baseband fading only.** The literature's practical choice is complex fading with Gaussian-integer lattices (Liu & Ling 2016) — two real dimensions align far better than one. This tier is the real-valued case.
- **N is small.** The exhaustive search is exponential in N; at toy scale that's fine. The LLL path is the scalable one.

---

## The math, made legible

### The problem: what combination is decodable?

The channel is `y = Σ h_i x_i + z` with `h` drawn i.i.d. The receiver scales by `α` and replays the shared dithers at the effective gains `α·h_i`:

```
y' = α·y + Σ α·h_i·d_i  =  Σ α·h_i·v_i + L·(integers) + α·z
```

Mod-L reduction kills the lattice components. What remains is the **real number** `Σ α·h_i·w_i + α·z` — and the decoder rounds it to the nearest integer. That rounding recovers the combination `Σ a_i·w_i mod L` **provided `α·h_i` is close to the integers `a_i`**. The mismatch is the self-noise:

```
Z_eff(α, a) = α²σ² + P·Σ (α·h_i − a_i)²          P = L²/12 (the cell's power)
R(α, a)     = ½·log₂⁺( P / Z_eff )                bits per real channel use
```

### Two facts that make it tractable

**The optimal scaling is closed-form (MMSE):**

```
α* = SNR·(hᵀa) / (1 + SNR·‖h‖²)
```

**Substituting α\* turns it into a shortest-lattice-vector problem:**

```
minimize D(a) = aᵀ(I + SNR·h·hᵀ)⁻¹·a   over integer a ≠ 0
```

The normalization pin: `N = 1, h = 1, a = 1` gives `D = 1/(1+SNR)` and `R = ½·log₂(1 + SNR)` — the AWGN capacity of a real channel, exactly. (The test suite asserts this to 1e-12.)

### Why `α = 1` is not optimal — and why Tier 1.5 got away with it

Tier 1.5 fixed `α = 1` and set `h_i = a_i`. In that case the self-noise term `Σ (α·h_i − a_i)² = 0` **identically** — the decode is exact by construction, not by tuning. That is a special case, and it is worth seeing how special:

```
h = [1, 2],  a = [1, 2]:   α* = 0.9405   (< 1 — even here!)
    rate(α*)   = 0.8747
    rate(α = 1) = 0.8305        ← what Tier 1.5 effectively assumes
    rate(a = [0, 1]) = 1.0070   ← what selection finds instead
```

Even an integer-aligned channel cannot align perfectly once `N ≥ 2`, because `α < 1` shrinks every gain and the residual `‖α·h − a‖²` is nonzero. The optimal response on this channel is to **drop the weaker node** — compute from `h₂ = 2` alone, at rate 1.007 bits/use, better than any 2-node combination. That is a real finding, not a bug: it is why the field treats coefficient selection as an optimization problem rather than a rounding.

---

## The code

### One fading channel — the problem in the open

```python
import numpy as np
from rf_compute.coefficients import select_coefficients, computation_rate, norm_bound, fading_gains

rng = np.random.default_rng(3)
h = fading_gains(3, rng)          # h_i ~ N(0, 1)

print(f"gains: {h}")               # e.g. [+2.041, -2.556, +0.418]
print(f"norm bound ||a|| <= {norm_bound(h, 5.0):.2f}")

a, rate = select_coefficients(h, 5.0, method="exhaustive")
print(f"selected: a = {list(a)}, rate = {rate:.4f}")   # [1, -1, 0], 1.58

print(f"plain sum: rate = {computation_rate(h, np.ones(3, dtype=int), 5.0):.4f}")  # 0.0000
```

The plain sum has **rate zero** on this channel: nothing can decode it. The selected combination `[1, −1, 0]` — node 1 minus node 2, node 3 dropped — is decodable at 1.58 bits/use.

### The three methods

```python
for method in ["exhaustive", "lll", "rounded"]:
    a, rate = select_coefficients(h, 5.0, method=method)
    print(f"{method:>11}: a = {list(a)}, rate = {rate:.4f}")
```

| method | what it does | complexity | lineage |
|---|---|---|---|
| `exhaustive` | searches the integer norm ball `‖a‖ ≤ √(1+SNR‖h‖²)` | exponential in N | Nazer & Gastpar 2011 (the bound is theirs) |
| `lll` | LLL-reduces the SLV lattice, enumerates small coordinates | polynomial | Sahraei & Gastpar 2014; Liu & Ling 2016 |
| `rounded` | nearest integer `round(α·h)`, iterated | linear | the baseline — what you get without optimizing |

### The kernel surface

```python
from rf_compute import FadingAirCompOperator, WaveComputeKernel

kernel = WaveComputeKernel(backend="sim")
op = FadingAirCompOperator(num_nodes=3, modulus=16, dimension=32, snr_db=5.0,
                           selection="exhaustive")
out = kernel.apply(op, 3)     # seed for reproducibility

print(out['gains'])                              # the realized fading
print(out['selected']['coefficients'],           # [1, -1, 0] — chosen
      out['selected']['rate'],                   # 1.5827 bits/use
      out['selected']['power_factor'])           # 0.13x baseline
print(out['plain']['rate'],                      # 0.0 — not decodable
      out['plain']['power_factor'])              # 2.04x baseline
```

### The full experiment (what the example runs)

```bash
python examples/tier1_6_fading_coefficients.py
```

---

## What you should see

### Selection vs. the plain sum over 800 fading channels

At 5 dB, 3 nodes, `L=16`, `n=32`, 800 independent realizations:

| strategy | mean rate (bits/use) | median power factor | mean power factor |
|---|---:|---:|---:|
| **selected** | **0.8258** | **0.39** | 0.67 |
| plain sum | 0.0194 | 6.27 | 2765 |

- **Plain-sum rate is ZERO on 94.6% of channels.** In the real (non-inverted) system that equation cannot be decoded at all — the receiver would have to fall back to routing.
- **Selection returns the plain sum on 1.0% of channels** — when the fading happens to align, `a = 1` really is optimal. Selection is earned, not assumed.
- **The median power factor is 0.39 for selection vs 6.27 for the plain sum.** Selection *drops* faded nodes instead of inverting them. The mean (2765×) is dominated by rare deep fades — read the median.

### Read the rate column, not the error column

In this inverted toy **both strategies decode at the same ~30% error threshold** — the pre-code re-aligns the integer gains, so fading does not add decode error here. That is worth stating plainly rather than hiding: the fading tier's visible wins are the **rate** (decodability) and the **power**, not the error rate. Reproducing the error-rate win requires the un-inverted `α < 1` receiver with an R^n lattice decoder — the honest gap between this toy and the theory.

---

## Troubleshooting (the honest list)

| Symptom | Cause | Fix |
|---|---|---|
| Selected rate is 0 | Every candidate in the norm ball is misaligned | Rare; check `norm_bound` — a tiny SNR or `‖h‖` shrinks the search ball to zero width. Raise SNR or accept that this channel is undecodable. |
| `a` has zeros | **Correct behavior** — selection dropped a faded node | Nothing to fix. `a_i = 0` means "don't compute from node i"; the power factor drops with it. |
| Exhaustive is slow at N ≥ 6 | The search is `(2r+1)^N`; the code caps radius to stay tractable | Use `method="lll"` — that is what it's for. |
| Mean power factor looks absurd (thousands) | Deep fades: `1/h_i²` explodes as `h_i → 0` | Read the **median**; the mean is not a robust statistic for this quantity. |
| Selection underperforms `rounded` | Should not happen (exhaustive ⊇ rounded's candidates) | If it does, report it — the test suite asserts exhaustive ≥ rounded across 25 random channels. |
| Results vary between runs | Fresh `default_rng()` when no seed given | Pass an int seed to `kernel.apply(op, seed)` or an `rng` to `fading_trial`. |

---

## What this proves

1. **The compute-and-forward coefficient vector is an optimization, not an arithmetic step.** Rounding `α·h` is a heuristic and it is not always optimal — on `h = [1, 2]` the optimum drops a node entirely.

2. **On a fading channel, the plain sum is usually not a decodable equation.** 94.6% of channels have rate zero for `a = 1`. This is the real reason the literature spends decades on coefficient selection: not error-rate tuning, *decodability*.

3. **Selection is also a power strategy.** Dropping faded nodes (median 0.39× baseline) beats inverting them (6.27×) — and the gap widens as the fade deepens.

4. **The theory's rate formula validates against a known limit.** With `N = 1, h = 1, a = 1` it returns exactly `½log₂(1+SNR)`, the AWGN capacity. The machinery is not a fit; it is the closed form, and the pin is a test.

---

## Going deeper

### The genuine open gap: no maintained OSS compute-and-forward implementation

A live literature scan (2026-09) found **no maintained open-source reference implementation of compute-and-forward coefficient selection** in any language. The algorithms are published (Sahraei & Gastpar's exact polynomial method, Liu & Ling's efficient search) but not packaged. Adjacent code exists — [`amazon-science/LatticeAlgorithms.jl`](https://github.com/amazon-science/LatticeAlgorithms.jl) (Julia, closest-lattice-point decoding, not compute-and-forward) and [NestQuant](https://arxiv.org/abs/2502.09720) (ICML 2025, dithered nested lattices for LLM quantization) — but the coefficient-selection problem itself has no reference implementation to point at. This module is a start at toy scale; a proper implementation (complex channels, Gaussian-integer lattices, LLL + Schnorr–Euchner enumeration) is a real contribution target.

### Toward complex channels

Real fading is the single-dimension case. Complex fading gives two real dimensions and the Gaussian-integer (or Eisenstein-integer) lattice, which aligns far better — the literature's practical choice (Liu & Ling 2016). The natural next tier.

### Toward the un-inverted receiver

The honest gap named in the scope section: a receiver that scales by real `α < 1` and decodes in R^n with a lattice decoder. That is where fading turns into decode error rather than just rate loss.

---

## Safety and legality

Entirely in simulation — no RF transmission, nothing to license. The Tier 1 rules apply if you proceed to hardware: ISM bands only, low power, check local regulations. See [hello-world-aircomp.md](hello-world-aircomp.md#safety-and-legality).

---

## Provenance

- **Founding result:** Nazer & Gastpar, "Compute-and-forward: Harnessing interference through structured codes," *IEEE Trans. Inf. Theory* 57(10), 6463–6486 (2011). [DOI: 10.1109/TIT.2011.2165816](https://doi.org/10.1109/TIT.2011.2165816)
- **Exact polynomial algorithm:** Sahraei & Gastpar, "Compute-and-forward: Finding the best equation," 52nd Allerton Conf. (2014). [arXiv:1410.3656](https://arxiv.org/abs/1410.3656)
- **Efficient integer search:** Liu & Ling, "Efficient Integer Coefficient Search for Compute-and-Forward," *IEEE Trans. Wireless Commun.* 15(12), 8039–8050 (2016). [arXiv:1609.05490](https://arxiv.org/abs/1609.05490)
- **MMSE scaling & rate formula:** Huang & Burr, "Low Complexity Coefficient Selection Algorithms for Compute-and-Forward," (2017). [arXiv:1704.05007](https://arxiv.org/abs/1704.05007)
- **Adjacent code:** [amazon-science/LatticeAlgorithms.jl](https://github.com/amazon-science/LatticeAlgorithms.jl); Savkin et al., "NestQuant: Nested Lattice Quantization for Matrix Products and LLMs," ICML 2025. [arXiv:2502.09720](https://arxiv.org/abs/2502.09720)
- **Module:** `rf_compute/coefficients.py` · **Fading trial:** `rf_compute/lattice.py` (`fading_trial`, `fading_scoreline`) · **Operator:** `FadingAirCompOperator` · **Example:** `examples/tier1_6_fading_coefficients.py` · **Tests:** `tests/test_coefficients.py`
- **Honest scope:** Simulated real baseband fading; channel inversion keeps the toy decoder exact; the rate numbers are the theory's, computed through the closed form validated against the single-user capacity pin. The un-inverted `α < 1` receiver with an R^n lattice decoder is NOT implemented — that is the boundary between this toy and the theorem.
