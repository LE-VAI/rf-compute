# rf-compute

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

## The $350 hello world

You don't need a fab, a clean room, or a metamaterial. You need **two transmit SDRs, one receive SDR, and a laptop.** But you can also start for free — the simulation kernel runs on NumPy alone.

```
Tx1 ──┐
       ├── air ──→ Rx ──→ f(x1 + x2)  ← the channel computed the sum
Tx2 ──┘
```

Two transmitters send pre-coded signals simultaneously. The receiver reads their sum from the superposed waveform — **without decoding either signal individually.** Interference is the operation. This is the Nazer & Gastpar 2007 result, made legible.

This is the simplest wave-compute primitive that exists. If you can run this, you understand the field.

📖 **Full walkthrough:** [`docs/hello-world-aircomp.md`](docs/hello-world-aircomp.md) — hardware, code, what to expect, troubleshooting

**Budget option (~$190):** One HackRF + one RTL-SDR. You transmit x1, x2, and x1+x2 sequentially and compare captures in post-processing. You lose the simultaneity that makes AirComp profound, but you learn the signal-processing structure for half the cost. Details in the walkthrough.

### Quick start (60 seconds, no hardware)

```bash
git clone https://github.com/rf-compute/rf-compute.git
cd rf-compute
pip install -e .
python examples/tier1_aircomp_kernel.py    # AirComp: 3+5=8
python examples/tier2_convolution_kernel.py # 4 operators
python examples/tier3_inversion_kernel.py   # solves Ax=b
```

The simulation kernel runs the same API as the hardware kernel — switch `backend="sim"` to `backend="sdr"` when you have SDRs. See [`docs/INSTALL.md`](docs/INSTALL.md) for the SDR driver install (the one friction point when you're ready for hardware).

---

## The hello-world ladder

Three reproducible experiments, escalating in cost. Each maps to a peer-reviewed result.

| Tier | Experiment | Cost | Proves | Citation |
|---|---|---|---|---|
| **1** | AirComp sum | ~$350 | Interference IS computation | Nazer & Gastpar, *IEEE TIT* (2011) |
| **2** | Wave-domain convolution | ~$200 | Linear operators are native to wave physics | Silva et al., *Science* (2014) |
| **3** | Matrix inversion via feedback | free / ~$200 / ~$350 | The wave domain solves equations; settling, not iterating | *Nature Communications* (2025) |

Tier 3 has three modes (simulation free, software-in-the-loop ~$200, analog feedback ~$350) — see the walkthrough for the trade-off.

📖 **Full ladder walkthroughs:** [Tier 1](docs/hello-world-aircomp.md) · [Tier 2](docs/hello-world-convolution.md) · [Tier 3](docs/hello-world-matrix-inversion.md) — each with parts lists, code, and troubleshooting

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

📖 **Full deferral table:** [`docs/field-map.md`](docs/field-map.md#what-we-are-not-building-graceful-deferral-to-the-community)

---

## The field map

The complete lineage-to-primitive map lives in [`docs/field-map.md`](docs/field-map.md). It covers:

- Each of the four lineages with origin papers, SOTA devices, key labs, and SDR-mappable primitives
- The SDR bridge: why software-defined radio is the unified developer surface
- The "hello world" ladder with cost estimates and parts lists
- Field maturity at a glance
- Full provenance and citation list

If you read one document, read the field map.

---

## Hardware you'll need

| Item | Tier 1 | Tier 2 | Tier 3 (Mode B/C) | Est. cost |
|---|:---:|:---:|:---:|---|
| HackRF One SDR (Tx) | ×2 | ×1 | ×1 | ~$150 each |
| RTL-SDR (Rx, receive-only) | ×1 | ×1 | ×1 | ~$30 |
| 10 MHz clock sync cable (BNC) | ✓ | | | ~$5 |
| Laptop (any OS) | ✓ | ✓ | ✓ | you have one |
| Passive scatterer / reflector | | ✓ (Mode B) | | ~$10–20 |
| RF circulator (one-way loop) | | | ✓ (Mode C) | ~$40 |
| Programmable attenuator (loop gain) | | | ✓ (Mode C) | ~$30 |
| RF splitter/combiner | | | ✓ (Mode C) | ~$15 |

**Total entry cost: free (Tier 3 sim) / ~$200 (Tier 2 or Tier 3 Mode B) / ~$350 (Tier 1 or Tier 3 Mode C).** No fab. No clean room. No metamaterial. Tier 3 Mode A is pure simulation and costs nothing — start there to see the math before buying hardware.

---

## Who this is for

- **Builders** who want to touch wave-domain compute without a physics PhD
- **Researchers** who want a shared vocabulary across the four lineages
- **Educators** who want reproducible experiments with real citations
- **Curious engineers** who read "the channel is the adder" and want to see it work

If you've never heard of RF-as-compute and want to understand it: start with the $350 hello world. If you're already in one of the four lineages: the field map is the bridge to the other three.

---

## The key papers (start here)

| Paper | Year | Why it matters |
|---|---|---|
| Nazer & Gastpar, "Computation over multiple-access channels," *IEEE Trans. Inf. Theory* | 2007 | The founding result. Interference computes functions. |
| Silva et al., "Performing mathematical operations with metamaterials," *Science* | 2014 | Passive materials do math on wavefields. |
| Torrejon et al., "Neuromorphic computing with spintronic oscillators," *Nature* | 2017 | Microwave nano-oscillators as neurons. 99.6% spoken-digit. |
| Zangeneh-Nejad et al., "Analogue computing with metamaterials," *Nature Reviews Materials* | 2021 | The canonical review. "Wave-based analog computing." |
| Li et al., "Performing calculus with ENZ metamaterials," *Science Advances* | 2022 | Differentiation + integration in the material. |
| "Programmable wave-based analog computing metastructure," *Nature Communications* | 2025 | **The SOTA.** Matrix inversion, Newton's method, Lagrangian optimization at 45 MHz. |
| Guan & Yao, "Microwave photonic neural network," *J. Lightwave Technology* | 2025 | RF photonic MVM. 55×10⁶ MAC/s. |

📖 **Annotated bibliography:** [`docs/bibliography.md`](docs/bibliography.md) — every citation, reading order, how to use it

---

## Status

**Pre-release.** The spine is complete: field map, three-tier hello-world ladder, annotated bibliography, contribution guide, and the SDR kernel abstraction.

- ✅ [Field map](docs/field-map.md) — the spine document
- ✅ [Tier 1: AirComp sum](docs/hello-world-aircomp.md) — the $350 hello world
- ✅ [Tier 2: Wave-domain convolution](docs/hello-world-convolution.md) — the $200 linear-operator primitive
- ✅ [Tier 3: Matrix inversion via feedback](docs/hello-world-matrix-inversion.md) — the capstone (free / $200 / $350)
- ✅ [Annotated bibliography](docs/bibliography.md) — every citation, reading order, how to use it
- ✅ [Installation guide](docs/INSTALL.md) — pip install, SDR drivers, troubleshooting
- ✅ [Contributing guide](CONTRIBUTING.md) — the reproducibility + provenance + honesty bar
- ✅ [LICENSE](LICENSE) — MIT
- ✅ [SDR kernel abstraction](rf_compute/) — one API across all four lineages
  - ✅ [`examples/tier1_aircomp_kernel.py`](examples/tier1_aircomp_kernel.py) — Tier 1 with the kernel
  - ✅ [`examples/tier2_convolution_kernel.py`](examples/tier2_convolution_kernel.py) — Tier 2 with the kernel
  - ✅ [`examples/tier3_inversion_kernel.py`](examples/tier3_inversion_kernel.py) — Tier 3 with the kernel
- ⏳ Community contributions (see `CONTRIBUTING.md`)

---

## License

MIT — see [LICENSE](LICENSE).

---

## Provenance

This project is a research-and-education surface synthesized from peer-reviewed literature. The underlying research was conducted 2026-07-31 via a multi-source academic search across *Science*, *Nature*, *Nature Photonics*, *IEEE Trans. Inf. Theory*, *Journal of Lightwave Technology*, and arXiv. Full provenance and citation verification in [`docs/field-map.md`](docs/field-map.md#provenance).

All citations are peer-reviewed unless marked `[preprint]` or `[vendor]`. Lightmatter performance figures are vendor-sourced. AirComp surveys are preprints. The spin-torque SDR mapping is approximate, not a hardware equivalent.

This surface does not claim authorship of the underlying physics. It claims only the bridge — the map, the vocabulary, and the reproducible "hello world."