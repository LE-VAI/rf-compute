# Hello World: Over-the-Air Federated Learning

## Tier 4 — The adder goes to work

> **The experiment:** 20 devices train one shared model. Every round, each computes a gradient on its own data and all of them transmit **at once**. The channel adds the gradients, and the server reads the sum. Over 60 rounds that costs 3,180 channel uses; giving each device its own slot costs 24,000, and both reach the same accuracy. Then the hardware problem arrives: the devices' signals land out of phase and out of step. With no phase sync, the naive receiver turns training into a coin flip (mean accuracy 0.47, range 0.25–0.77 over 10 seeds). A shared pilot and a two-tap equalizer restore it to 0.84, without the receiver ever learning a single device's offset or gradient.

Tiers 1–1.6 built the primitive: the channel is an adder. This tier puts it to work on the job it is most often proposed for, **aggregating model updates in federated learning**. It is the capstone of the AirComp lineage: Tier 1's superposition, Tier 1.6's lesson that deep fades should be dropped rather than inverted, and a new one — what misalignment does to the sum, and what a receiver can do about it.

It maps to the over-the-air federated learning literature: Zhu, Wang & Huang's broadband analog aggregation (*IEEE Trans. Wireless Commun.* 19(1), 491–506, 2020), Yang, Jiang, Shi & Ding (*IEEE Trans. Wireless Commun.* 19(3), 2022–2035, 2020), and Amiri & Gündüz's distributed SGD over the air (*IEEE Trans. Signal Process.* 68, 2155–2169, 2020). The misalignment model follows Shao, Gündüz & Liew (*IEEE Trans. Wireless Commun.* 21(6), 3951–3964, 2022).

**Cost:** $0. Pure NumPy, no hardware, about 10 seconds to run.

---

## Honest scope

What this walkthrough demonstrates, precisely:

- **The channel-use win.** One over-the-air aggregation costs `d + 1` samples plus a 32-symbol pilot, whether 5 devices transmit or 50. Orthogonal slots cost `K·d`. At 20 devices and 20 features, that is 3,180 against 24,000 channel uses over the run, at the same final accuracy.
- **What misalignment does.** Residual phase offsets rotate each device's contribution; timing offsets smear each symbol into the next. Moderate misalignment ruins the *sum* (aggregation NMSE about 0.15) but not the *learning*. With no phase sync at all, the naive receiver's expected estimate is zero.
- **What a pilot-aided equalizer recovers, and what it can't.** The receiver measures only the aggregate channel, and it is exact only for what the devices' gradients have **in common**. Its error grows with how different the devices' data are.

What this walkthrough does NOT do (the honest list):

- **No hardware.** Simulated block fading. The signal structure is complete; the RF part is Tier 1's setup.
- **A one-sample-per-symbol receiver.** Shao, Gündüz & Liew oversample with a whitened matched filter and decode with a sum-product ML estimator or an aligned-sample estimator. Those are strictly stronger receivers, and their official code is MIT-licensed ([hku-icl/MisAlignedOAC](https://github.com/hku-icl/MisAlignedOAC)). The equalizer here is the one-screen version that shows *why* a receiver has to know about misalignment at all.
- **A small convex model.** Logistic regression on synthetic data, one gradient step per device per round (federated SGD). It isn't an FL library, and deep networks, local epochs and client sampling are all out of scope.
- **The orthogonal baseline is error-free.** It's an upper bound on accuracy, not a like-for-like noisy orthogonal link, which would do worse.
- **Real-valued gradients over a complex channel**, rectangular pulses, and offsets drawn fresh every round.

---

## The math, made legible

### One round, aligned

Device `k` holds gradient `g_k ∈ R^d`, clipped to `‖g_k‖ ≤ C`. The channel is block fading, `h_k ~ CN(0, 1)`. Devices with `|h_k|` below a truncation threshold sit the round out (Tier 1.6's lesson: don't pay to invert a deep fade). Each participant pre-codes

```
x_k = √η · g_k · conj(h_k) / |h_k|²
```

so the channel delivers exactly `√η · g_k`. The scaling `η = P·d·min|h_k|² / C²` is the largest one that keeps **every** participant within its per-symbol power budget `P`, so the weakest channel sets the scale for everyone. The receiver sees

```
r = √η · Σ_k g_k + z
```

and divides by `√η` and the participant count. The sum costs `d` channel uses (plus one sample of tail, below) no matter how many devices transmit. That is the whole point.

### What misalignment does

Real devices miss the alignment two ways:

- **Phase.** Channel-inversion pre-coding removes the channel phase only as well as the channel estimate allows. Device `k` arrives rotated by a residual `φ_k`.
- **Timing.** Device `k`'s symbols arrive `τ_k` of a symbol late. With rectangular pulses, receiver sample `j` holds `(1 − τ_k)` of symbol `j` and `τ_k` of symbol `j − 1`: inter-symbol interference. That is why the receiver takes `d + 1` samples, since the last one holds the tail.

```
r_j = √η · Σ_k e^{iφ_k} [ (1 − τ_k) g_k[j] + τ_k g_k[j−1] ] + z_j
```

The **naive receiver** reads `Re(r_j)` as if nothing had happened. With `φ_k ~ U(−π, π)` (no phase sync), `E[cos φ_k] = 0`, so the naive estimate's expected value is **zero**. What's left is noise that happens to be correlated with the gradients.

### The equalizer: measure the aggregate, never the individual

All participants transmit the same known ±1 pilot at the same time. The receiver can then measure the two **aggregate** taps

```
c0 = Σ_k e^{iφ_k} (1 − τ_k)        c1 = Σ_k e^{iφ_k} τ_k
```

by least squares. It never learns any single device's `φ_k` or `τ_k`, and AirComp's structure is preserved. It then solves the two-tap deconvolution

```
r / √η  ≈  c0 · u[j] + c1 · u[j−1]
```

for a real vector `u` by least squares. The real and imaginary parts of the `d + 1` samples give `2(d + 1)` equations for `d` unknowns.

**Why this works, and where it stops.** If every device sent the same gradient `ḡ`, then `r/√η` would be *exactly* `c0·ḡ[j] + c1·ḡ[j−1]`, and `u = ḡ`. They don't: each `g_k = ḡ + δ_k`. The equalizer handles the common part `ḡ` exactly and leaves the deviations `δ_k` smeared by the offsets. **The equalizer's error scales with the gradient spread** `mean_k ‖g_k − ḡ‖² / ‖ḡ‖²`. A shared clock shrinks the offsets, but nothing at a one-sample receiver removes the heterogeneity term.

---

## The code

### One aggregation through the kernel

```python
import numpy as np
from rf_compute import OTAAggregationOperator, WaveComputeKernel

kernel = WaveComputeKernel(backend="sim")
grads = [...]   # K gradient vectors, all the same length

op = OTAAggregationOperator(snr_db=20.0, max_phase=np.pi / 3, max_timing=0.5,
                            receiver="equalized", seed=1)
out = kernel.apply(op, inputs=grads)
out["estimate"]        # the receiver's estimate of the average gradient
out["nmse"]            # error vs the true average, normalized
out["max_tx_power"]    # the peak per-symbol power any device used (<= 1)
out["channel_uses"]    # d + 1 + pilot length, independent of K
```

### Federated training under one scheme

```python
from rf_compute import ota_fl

devices, test, _ = ota_fl.make_federated_data(num_devices=20, dim=20,
                                              heterogeneity=2.0,
                                              rng=np.random.default_rng(7))
hist = ota_fl.train("ota_equalized", devices=devices, test=test, rounds=60,
                    snr_db=20.0, max_phase=np.pi, max_timing=1.0,
                    rng=np.random.default_rng(8))
hist["test_accuracy"][-1]
```

The four schemes are `orthogonal` (error-free, `K·d` uses per round), `ota_aligned`, `ota_misaligned` (naive receiver), and `ota_equalized`.

### The full experiment (what the example runs)

```bash
python examples/tier4_ota_federated_learning.py
```

`ota_fl.scoreline(...)` trains every scheme on the same data with the same channel randomness, over 10 seeds, and reports mean, min and max accuracy. `ota_fl.aggregation_quality(...)` measures one aggregation's error at a fixed point over 400 channel draws.

---

## What you should see

### Aggregation error at a fixed point

20 devices, 20 dB, moderate misalignment (`φ ≤ 60°`, `τ ≤ 0.5`), 400 channel draws at the zero start:

| gradient spread | aligned | misaligned, naive | misaligned, equalized |
|---:|---:|---:|---:|
| 0.20 | 0.0009 | 0.141 | **0.0040** |
| 0.61 | 0.0007 | 0.140 | **0.0069** |
| 2.46 | 0.0006 | 0.159 | **0.0201** |

The naive receiver's error is almost all bias, and it barely moves with the data. The equalizer cuts it by 8–35×, but its error grows fivefold as the gradient spread grows twelvefold: the heterogeneity term, measured.

### Training: 20 devices, 60 rounds, 20 dB, gradient spread 0.61, 10 seeds

| scheme | moderate misalignment | no phase sync, `τ ≤ 1` | channel uses |
|---|---:|---:|---:|
| orthogonal, error-free (upper bound) | 0.842 (0.84–0.85) | 0.842 (0.84–0.85) | 24,000 |
| over the air, aligned | 0.842 (0.84–0.85) | 0.842 (0.84–0.85) | 3,180 |
| over the air, misaligned, naive | 0.839 (0.83–0.85) | **0.474 (0.25–0.77)** | 3,180 |
| over the air, misaligned, equalized | 0.842 (0.83–0.85) | **0.840 (0.83–0.85)** | 3,180 |

Final test accuracy: mean (min–max) over 10 seeds.

### Read the spread, not one run

Report the min–max, never a single seed. With no phase sync, one lucky seed of the naive receiver reaches 0.77 and looks almost fine. Over 10 seeds it averages 0.47, and its worst run is *worse than chance* (0.25): the model walked the wrong way. The seed-to-seed spread is the finding.

### Read the accuracy, not only the aggregation error

In the no-phase-sync case at high heterogeneity (spread 2.5), the equalizer's per-round NMSE is about 1.7: any single aggregation is worse than guessing zero. Yet its training still matches the upper bound (0.866 vs 0.867). The error changes every round, because the offsets are fresh, so SGD averages it out. The naive receiver's failure is different in kind: its *signal* averages to zero. Aggregation error and learning outcome are different questions, and this tier measures both.

---

## Troubleshooting (the honest list)

| Symptom | Cause | Fix |
|---|---|---|
| Naive receiver trains fine under misalignment | Moderate offsets shrink the gradient consistently; that acts like a smaller learning rate | **Correct behavior.** Try `max_phase=np.pi` (no phase sync) to see it fail. |
| Equalizer NMSE above 1, yet training works | The error is fresh every round, so SGD averages it | Read the accuracy column. A consistent bias would not average out. |
| One run looks great, another terrible | Naive receiver under no phase sync is a coin flip | Use `scoreline(num_seeds=10)` and read min–max. |
| Equalizer error grows with `heterogeneity` | The equalizer is exact only for the common part of the gradients | **That is the lesson.** Lower the spread or raise SNR; a stronger receiver (Shao et al.'s oversampled ML) does better. |
| Participation below K | Truncation: devices with `|h| < 0.3` sit the round out | Lower `truncation`. The weakest participant sets `η`, so the noise floor rises with it. |
| `max_tx_power` well below 1 | Gradients much smaller than the clip `C` waste the power budget | Tighten `clip` toward the typical gradient norm. |

---

## What this proves

1. **Over the air, aggregation cost is independent of the number of devices.** `d + 1 + pilot` channel uses per round for 20 devices or 2,000. Orthogonal slots scale with `K`. That is the whole case for AirComp in federated learning.

2. **Misalignment is a receiver problem as much as a hardware problem.** With no phase sync the naive receiver's expected estimate is zero, and training is a coin flip. A shared pilot plus a two-tap equalizer recovers upper-bound accuracy using only aggregate channel knowledge.

3. **What the equalizer can't fix is heterogeneity.** It is exact for what the devices share, and its error grows with the spread of their gradients. Sync hardware (Tier 1's 10 MHz clock cable) shrinks the offsets, but it does not make devices' data agree.

4. **Learning tolerates variance better than bias.** Moderate misalignment (a consistent shrink) and a noisy-but-fresh equalizer error both leave accuracy intact. A zero-mean signal does not.

---

## Going deeper

### Stronger receivers

Shao, Gündüz & Liew oversample the misaligned superposition with a whitened matched filter and estimate the sum with a sum-product ML estimator (linear in packet length) or an aligned-sample estimator. They report that at low SNR the aligned-sample estimator gives better test accuracy when phase misalignment is not severe, while the ML estimator suffers from error propagation and noise enhancement; at high SNR the ML estimator reaches optimal learning performance regardless of phase misalignment. Their official implementation is [hku-icl/MisAlignedOAC](https://github.com/hku-icl/MisAlignedOAC) (PyTorch, MIT). Porting the aligned-sample estimator to this kernel is a natural contribution.

### Digital AirComp: the next rung

Analog aggregation needs the signal to survive in amplitude. Digital schemes quantize first: one-bit aggregation with majority-vote decoding (Zhu, Du, Gündüz & Huang, *IEEE Trans. Wireless Commun.*, 2021, [DOI 10.1109/TWC.2020.3039309](https://doi.org/10.1109/TWC.2020.3039309)), type-based multiple access, and ChannelComp-style computational constellations. September 2026 alone brought several preprints in this line, and "The Computing Channel: How Modulation Programs the Airwaves" (Razavikia & Fischione, [arXiv:2609.11145](https://arxiv.org/abs/2609.11145)) is a tutorial-grade overview. It bridges this tier and the exact lattice arithmetic of Tier 1.5.

### Current directions

Over-the-air FL without assuming every device transmits or a known fading model: FedOAG (Xiang, Michelusi, Eldar & Su, MobiHoc 2026, [arXiv:2609.32832](https://arxiv.org/abs/2609.32832)).

---

## Safety and legality

Entirely in simulation — no RF transmission, nothing to license. The Tier 1 rules apply if you move to hardware: ISM bands only, low power, check local regulations. See [hello-world-aircomp.md](hello-world-aircomp.md#safety-and-legality).

---

## Provenance

- **Analog aggregation for federated edge learning:** Zhu, G., Wang, Y. & Huang, K. "Broadband Analog Aggregation for Low-Latency Federated Edge Learning." *IEEE Trans. Wireless Commun.* 19(1), 491–506 (2020). [DOI 10.1109/TWC.2019.2946245](https://doi.org/10.1109/TWC.2019.2946245)
- **FL via over-the-air computation:** Yang, K., Jiang, T., Shi, Y. & Ding, Z. "Federated Learning via Over-the-Air Computation." *IEEE Trans. Wireless Commun.* 19(3), 2022–2035 (2020). [DOI 10.1109/TWC.2019.2961673](https://doi.org/10.1109/TWC.2019.2961673)
- **Distributed SGD over the air:** Amiri, M. M. & Gündüz, D. "Machine Learning at the Wireless Edge: Distributed Stochastic Gradient Descent Over-the-Air." *IEEE Trans. Signal Process.* 68, 2155–2169 (2020). [DOI 10.1109/TSP.2020.2981904](https://doi.org/10.1109/TSP.2020.2981904)
- **Misalignment model:** Shao, Y., Gündüz, D. & Liew, S. C. "Federated Edge Learning With Misaligned Over-the-Air Computation." *IEEE Trans. Wireless Commun.* 21(6), 3951–3964 (2022). [DOI 10.1109/TWC.2021.3125798](https://doi.org/10.1109/TWC.2021.3125798) · [arXiv:2102.13604](https://arxiv.org/abs/2102.13604) · code: [hku-icl/MisAlignedOAC](https://github.com/hku-icl/MisAlignedOAC)
- **Citation check:** DOIs, volumes and pages verified against Crossref on 2026-09-30; the Shao et al. abstract was read from arXiv on the same date.
- **Module:** `rf_compute/ota_fl.py` · **Operator:** `OTAAggregationOperator` · **Example:** `examples/tier4_ota_federated_learning.py` · **Tests:** `tests/test_ota_fl.py`
- **Honest scope:** Simulated block fading, one sample per symbol, rectangular pulses, logistic regression on synthetic non-IID data. The orthogonal baseline is error-free, an upper bound. Every number above is reproducible from the example with its fixed seeds.
