# rf-compute

<!-- vai-hero:start -->
<p align="center">
  <picture>
    <source media="(prefers-reduced-motion: reduce)" srcset="https://raw.githubusercontent.com/LE-VAI/rf-compute/main/docs/media/hero-poster.png">
    <img src="https://raw.githubusercontent.com/LE-VAI/rf-compute/main/docs/media/hero-loop.webp" width="800" alt="Two carrier waves travel across the frame. Carrier A completes one cycle per loop and carrier B completes two. The bottom lane shows their sum, sampled as orange stems: the interference is the operation.">
  </picture>
</p>
<p align="center"><sub>A 4-second loop. It plays once and rests, and shows a still frame if you prefer reduced motion. <a href="https://le-vai.github.io/LE-VAI/loops/#rf-compute">Watch it on repeat</a>.</sub></p>
<!-- vai-hero:end -->

**Radio Frequency as a computational substrate — not a transmission medium.**

A research-and-education surface for wave-domain computation. The waveform is the operand. Interference isn't noise to cancel — it's the multiply-accumulate operation. The medium is the math.

---

## The one-sentence version

> Instead of using RF to carry bits to a digital chip that does the math, you make the RF waveform *be* the math — interference performs the operation, the channel is the adder, the medium is the algorithm.

This is not a new idea. It's been hiding in plain sight across four separate research communities for ~18 years. `rf-compute` is the bridge that makes it legible to builders.

---

## Why this exists

There is no unified developer surface for RF-as-compute. The field is real, peer-reviewed, and active — but it's scattered across four communities that don't share vocabulary, tooling, or a "hello world":

| Lineage | Where it publishes | What it does |
|---|---|---|
| **Computational metamaterials** | *Science*, *Nature* | Passive materials that perform math on wavefields |
| **Over-the-air computation (AirComp)** | *IEEE Trans. Inf. Theory* | The wireless channel *is* the computation |
| **Microwave photonics** | *Nature Photonics* | RF signals on optical carriers doing NN inference |
| **Spin-torque neuromorphic** | *Nature Electronics* | Microwave nano-oscillators as neurons |

They all share one thing: **the waveform is the operand.** Nobody has built the bridge between them. That's the gap this project fills.

---

## The <span>$</span>350 hello world

You don't need a fab, a clean room, or a metamaterial. You need **two transmit SDRs, one receive SDR, and a laptop** — clone-tier hardware gets you there around <span>$</span>350; official units run ~<span>$</span>725. Either way, the simulation kernel runs free on NumPy alone.

```
Tx1 ──┐
       ├── air ──→ Rx ──→ f(x1 + x2)  ← the channel computed the sum
Tx2 ──┘
```

Two transmitters send pre-coded signals simultaneously. The receiver reads their sum from the superposed waveform — **without decoding either signal individually.** Interference is the operation. This is the Nazer & Gastpar 2007 result, made legible.

This is the simplest wave-compute primitive that exists. If you can run this, you understand the field.

📖 **Full walkthrough:** [`docs/hello-world-aircomp.md`](https://github.com/LE-VAI/rf-compute/blob/main/docs/hello-world-aircomp.md) — hardware, code, what to expect, troubleshooting

**Budget option:** One HackRF + one RTL-SDR (clone-tier ~<span>$</span>190, official ~<span>$</span>380). You transmit x1, x2, and x1+x2 sequentially and compare captures in post-processing. You lose the simultaneity that makes AirComp profound, but you learn the signal-processing structure for half the cost. Details in the walkthrough.

### Quick start (60 seconds, no hardware)

```bash
git clone https://github.com/LE-VAI/rf-compute.git
cd rf-compute
pip install -e .
python examples/tier1_aircomp_kernel.py    # AirComp: 3+5=8
python examples/tier1_5_lattice_aircomp_kernel.py  # exact lattice sums, $0
python examples/tier1_6_fading_coefficients.py     # coefficient selection on fading, $0
python examples/tier2_convolution_kernel.py # 4 operators
python examples/tier3_inversion_kernel.py   # solves Ax=b
python examples/tier4_ota_federated_learning.py  # federated learning over the air, $0
```

The simulation kernel runs the same API as the hardware kernel — switch `backend="sim"` to `backend="sdr"` when you have SDRs. See [`docs/INSTALL.md`](https://github.com/LE-VAI/rf-compute/blob/main/docs/INSTALL.md) for the SDR driver install (the one friction point when you're ready for hardware).

---

## The hello-world ladder

Six reproducible experiments, escalating in cost. Each maps to a peer-reviewed result.

| Tier | Experiment | Cost (clone-tier / official) | Proves | Citation |
|---|---|---|---|---|
| **1** | AirComp sum | ~<span>$</span>350 / ~<span>$</span>725 | Interference IS computation | Nazer & Gastpar, *IEEE TIT* (2011) |
| **1.5** | Lattice-coded AirComp | free (sim) | The exact result: noise-resilient sums in one channel use, no message decoded | Nazer & Gastpar, *IEEE TIT* (2007/2011) |
| **1.6** | Fading-channel coefficients | free (sim) | Coefficients are an optimization: the plain sum is undecodable on ~95% of fading channels; selection finds a decodable one | Nazer & Gastpar (2011); Sahraei & Gastpar (2014); Liu & Ling, *IEEE TWC* (2016) |
| **2** | Wave-domain convolution | ~<span>$</span>200 / ~<span>$</span>395 | Linear operators are native to wave physics | Silva et al., *Science* (2014) |
| **3** | Matrix inversion via feedback | free / ~<span>$</span>200 / ~<span>$</span>350 | The wave domain solves equations; settling, not iterating | Tzarouchis, Edwards & Engheta, *Nature Communications* (2025) |
| **4** | Over-the-air federated learning | free (sim) | The channel aggregates model updates in `d+1` uses however many devices transmit; a shared pilot rescues training from misalignment, and heterogeneity bounds the rescue | Zhu, Wang & Huang, *IEEE TWC* (2020); Shao, Gündüz & Liew, *IEEE TWC* (2022) |

Tier 3 has three modes (simulation free, software-in-the-loop ~<span>$</span>200 clone / ~<span>$</span>395 official, analog feedback ~<span>$</span>350 clone / ~<span>$</span>480 official) — see the walkthrough for the trade-off.

📖 **Full ladder walkthroughs:** [Tier 1](https://github.com/LE-VAI/rf-compute/blob/main/docs/hello-world-aircomp.md) · [Tier 1.5](https://github.com/LE-VAI/rf-compute/blob/main/docs/hello-world-lattice-aircomp.md) · [Tier 1.6](https://github.com/LE-VAI/rf-compute/blob/main/docs/hello-world-fading-coefficients.md) · [Tier 2](https://github.com/LE-VAI/rf-compute/blob/main/docs/hello-world-convolution.md) · [Tier 3](https://github.com/LE-VAI/rf-compute/blob/main/docs/hello-world-matrix-inversion.md) · [Tier 4](https://github.com/LE-VAI/rf-compute/blob/main/docs/hello-world-ota-federated-learning.md) · [Tier 4-h](https://github.com/LE-VAI/rf-compute/blob/main/docs/hello-world-hardware-reality.md) — each with parts lists, code, and troubleshooting

---

## What this is NOT

This is a **research and education surface**, not a product. We build the map, the vocabulary, and the reproducible "hello world." The community builds the future.

We are explicitly **not** building:

- ❌ Deep-learning-scale matrices (SOTA is 5×5; NN layers are 1024×1024 — that's a hardware physics problem)
- ❌ Phone-form-factor integration (45 MHz ≈ 6.7m wavelength — subwavelength resonator design at phone scale is unsolved)
- ❌ Noise-resilient analog compute for hostile EM environments (lab results won't hold in a pocket)
- ❌ 6G standards integration (AirComp is a 6G research candidate, not yet in 3GPP standardization; that's a multi-year institutional process)
- ❌ Commercial fabrication (Lightmatter is pursuing the optical frontier; pure-RF commercial compute is not this project)

These are real, important problems. They belong to the metamaterials physics community, the antenna engineers, the RF circuit designers, the standards bodies, and industry — not to a research-and-education surface. We name them clearly so builders know where the open frontier lives.

📖 **Full deferral table:** [`docs/field-map.md`](https://github.com/LE-VAI/rf-compute/blob/main/docs/field-map.md#what-we-are-not-building-graceful-deferral-to-the-community)

---

## The field map

The complete lineage-to-primitive map lives in [`docs/field-map.md`](https://github.com/LE-VAI/rf-compute/blob/main/docs/field-map.md). It covers:

- Each of the four lineages with origin papers, SOTA devices, key labs, and SDR-mappable primitives
- The SDR bridge: why software-defined radio is the unified developer surface
- The "hello world" ladder with cost estimates and parts lists
- Field maturity at a glance
- Full provenance and citation list

If you read one document, read the field map.

---

## Hardware you'll need

Prices verified 2026-08-31, re-checked 2026-09-30. HackRF One official retail is ~<span>$</span>340 (SparkFun/Adafruit have retired the unit; GSG's successor **HackRF Pro** is ~<span>$</span>400). AliExpress clones run ~<span>$</span>100–150 but degrade above 1 GHz per Great Scott Gadgets' own clone test — fine for sub-GHz learning experiments, not for precision work. RTL-SDR Blog V4 is end-of-line (May 2026); current official units are the V3 or V4L at ~<span>$</span>35–40 (the V4L is itself a limited edition with roughly a year of chip stock).

| Item | Tier 1 | Tier 2 | Tier 3 (Mode B/C) | Official | Clone-tier |
|---|:---:|:---:|:---:|---|---|
| HackRF One SDR (Tx) | ×2 | ×1 | ×1 | ~<span>$</span>340 each | ~<span>$</span>100–150 each |
| RTL-SDR (Rx, receive-only) | ×1 | ×1 | ×1 | ~<span>$</span>35–40 | ~<span>$</span>30 |
| 10 MHz clock sync cable (BNC/SMA) | ✓ | | | ~<span>$</span>5 | — |
| Laptop (any OS) | ✓ | ✓ | ✓ | you have one | — |
| Passive scatterer / reflector | | ✓ (Mode B) | | ~<span>$</span>10–20 | DIY |
| RF circulator (one-way loop) | | | ✓ (Mode C) | ~<span>$</span>40 | — |
| Programmable attenuator (loop gain) | | | ✓ (Mode C) | ~<span>$</span>30 | — |
| RF splitter/combiner | | | ✓ (Mode C) | ~<span>$</span>15–25 | — |

**Total entry cost: free (Tier 3 sim) / clone-tier ~<span>$</span>200–350 / official ~<span>$</span>390–725 depending on tier.** No fab. No clean room. No metamaterial. Tier 3 Mode A is pure simulation and costs nothing — start there to see the math before buying hardware. If you buy clones, know what you're buying: they work for learning, they drift for precision.

---

## Who this is for

- **Builders** who want to touch wave-domain compute without a physics PhD
- **Researchers** who want a shared vocabulary across the four lineages
- **Educators** who want reproducible experiments with real citations
- **Curious engineers** who read "the channel is the adder" and want to see it work

If you've never heard of RF-as-compute and want to understand it: start with the <span>$</span>350 hello world. If you're already in one of the four lineages: the field map is the bridge to the other three.

---

## The key papers (start here)

| Paper | Year | Why it matters |
|---|---|---|
| Nazer & Gastpar, "Computation over multiple-access channels," *IEEE Trans. Inf. Theory* | 2007 | The founding result. Interference computes functions. |
| Silva et al., "Performing mathematical operations with metamaterials," *Science* | 2014 | Passive materials do math on wavefields. |
| Torrejon et al., "Neuromorphic computing with spintronic oscillators," *Nature* | 2017 | Microwave nano-oscillators as neurons. 99.6% spoken-digit. |
| Zangeneh-Nejad et al., "Analogue computing with metamaterials," *Nature Reviews Materials* | 2021 | The canonical review. "Wave-based analog computing." |
| Li et al., "Performing calculus with ENZ metamaterials," *Science Advances* | 2022 | Differentiation + integration in the material. |
| Tzarouchis, Edwards & Engheta, "Programmable wave-based analog computing machine: a metastructure that designs metastructures," *Nature Communications* 16, 908 ([DOI](https://doi.org/10.1038/s41467-025-56019-1)) | 2025 | **The SOTA.** Matrix inversion, Newton's method, Lagrangian optimization at 45 MHz. |
| Chegini, Guan & Yao, "Microwave photonic neural network," *J. Lightwave Technology* | 2025 | RF photonic MVM. 55×10⁶ MAC/s. |

📖 **Annotated bibliography:** [`docs/bibliography.md`](https://github.com/LE-VAI/rf-compute/blob/main/docs/bibliography.md) — every citation, reading order, how to use it

---

## Status

**Pre-release.** The spine is complete: field map, six-tier hello-world ladder, annotated bibliography, contribution guide, and the SDR kernel abstraction. The simulation kernel is fully reproducible (every run seed-deterministic; 116 tests).

- ✅ [Field map](https://github.com/LE-VAI/rf-compute/blob/main/docs/field-map.md) — the spine document
- ✅ [Tier 1: AirComp sum](https://github.com/LE-VAI/rf-compute/blob/main/docs/hello-world-aircomp.md) — the <span>$</span>350 hello world
- ✅ [Tier 1.5: Lattice-coded AirComp](https://github.com/LE-VAI/rf-compute/blob/main/docs/hello-world-lattice-aircomp.md) — the <span>$</span>0 exact-computation primitive (nested-lattice kernel)
- ✅ [Tier 1.6: Fading-channel coefficients](https://github.com/LE-VAI/rf-compute/blob/main/docs/hello-world-fading-coefficients.md) — the <span>$</span>0 coefficient-selection tier (computation-rate maximization, MMSE α, LLL + norm-bound search)
- ✅ [Tier 2: Wave-domain convolution](https://github.com/LE-VAI/rf-compute/blob/main/docs/hello-world-convolution.md) — the <span>$</span>200 linear-operator primitive
- ✅ [Tier 3: Matrix inversion via feedback](https://github.com/LE-VAI/rf-compute/blob/main/docs/hello-world-matrix-inversion.md) — the wave-domain capstone (free / <span>$</span>200 / <span>$</span>350)
- ✅ [Tier 4: Over-the-air federated learning](https://github.com/LE-VAI/rf-compute/blob/main/docs/hello-world-ota-federated-learning.md) — the AirComp capstone (<span>$</span>0): aggregation over the air, misalignment, and a pilot-aided equalizer
- ✅ **Tier 4-h: Hardware reality** (<span>$</span>0) — the same aggregation through a *real* receiver. Tier 4 models misalignment as a phase held constant across the block; a <span>$</span>2–25 ESP32-class board also has a carrier offset that **rotates**, a timing offset that **drifts**, an uncalibrated ADC path, and **burst** capture. The result: the pilot-aided equalizer does **not** survive an unsynchronised front end — and what breaks it is the oscillator, not the ADC. `DEV_BOARD_UNCALIBRATED` takes the median aggregation NMSE from **0.42 to 17.5**, while 10-bit quantization, a −30 dB IQ image and a 1 dB AGC error all stay within 25% of a clean receiver. The fix is a **second-order decision-directed PLL** (the tracking receiver), which clears **7 rad** of accumulated phase — 14× past the block fit's half-radian wall — provided the loop gain is *tuned* to `kp=0.8, ki=0.2`; a narrow loop fails the same way. What no single-phase receiver fixes is a **spread** of independent per-device offsets; a genie receiver handed the true combined channel shows that gap is mostly the receiver (genie 0.78 vs the tracker's 4.68) and partly the channel (the genie's own error rises from 0.45). `python examples/tier4h_hardware_reality.py`
- ✅ [Annotated bibliography](https://github.com/LE-VAI/rf-compute/blob/main/docs/bibliography.md) — every citation, reading order, how to use it
- ✅ [Installation guide](https://github.com/LE-VAI/rf-compute/blob/main/docs/INSTALL.md) — pip install, SDR drivers, troubleshooting
- ✅ [Contributing guide](https://github.com/LE-VAI/rf-compute/blob/main/CONTRIBUTING.md) — the reproducibility + provenance + honesty bar
- ✅ [LICENSE](https://github.com/LE-VAI/rf-compute/blob/main/LICENSE) — MIT
- ✅ [SDR kernel abstraction](https://github.com/LE-VAI/rf-compute/tree/main/rf_compute/) — one API across all four lineages
  - ✅ [`examples/tier1_aircomp_kernel.py`](https://github.com/LE-VAI/rf-compute/blob/main/examples/tier1_aircomp_kernel.py) — Tier 1 with the kernel
  - ✅ [`examples/tier1_5_lattice_aircomp_kernel.py`](https://github.com/LE-VAI/rf-compute/blob/main/examples/tier1_5_lattice_aircomp_kernel.py) — Tier 1.5 with the kernel
  - ✅ [`examples/tier1_6_fading_coefficients.py`](https://github.com/LE-VAI/rf-compute/blob/main/examples/tier1_6_fading_coefficients.py) — Tier 1.6 with the kernel
  - ✅ [`examples/tier2_convolution_kernel.py`](https://github.com/LE-VAI/rf-compute/blob/main/examples/tier2_convolution_kernel.py) — Tier 2 with the kernel
  - ✅ [`examples/tier3_inversion_kernel.py`](https://github.com/LE-VAI/rf-compute/blob/main/examples/tier3_inversion_kernel.py) — Tier 3 with the kernel
  - ✅ [`examples/tier4_ota_federated_learning.py`](https://github.com/LE-VAI/rf-compute/blob/main/examples/tier4_ota_federated_learning.py) — Tier 4 with the kernel
  - ✅ [`examples/tier4h_hardware_reality.py`](https://github.com/LE-VAI/rf-compute/blob/main/examples/tier4h_hardware_reality.py) — Tier 4-h: the aggregation through a real receiver (survival table, impairment attribution, the accumulated-phase ceiling)
- ⏳ Community contributions (see `CONTRIBUTING.md`)

---

## License

MIT — see [LICENSE](https://github.com/LE-VAI/rf-compute/blob/main/LICENSE).

---

## Provenance

This project is a research-and-education surface synthesized from peer-reviewed literature. The underlying research was conducted 2026-07-31 via a multi-source academic search across *Science*, *Nature*, *Nature Photonics*, *IEEE Trans. Inf. Theory*, *Journal of Lightwave Technology*, and arXiv. Full provenance and citation verification in [`docs/field-map.md`](https://github.com/LE-VAI/rf-compute/blob/main/docs/field-map.md#provenance).

All citations are peer-reviewed unless marked `[preprint]` or `[vendor]`. Lightmatter performance figures are vendor-sourced. AirComp surveys are preprints. The spin-torque SDR mapping is approximate, not a hardware equivalent.

This surface does not claim authorship of the underlying physics. It claims only the bridge — the map, the vocabulary, and the reproducible "hello world."