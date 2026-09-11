# RF Compute Field Map

## A research-and-education surface for wave-domain computation

**Date:** 2026-07-31
**Status:** Spine draft — determines whether the public GitHub preview has real structure or is vibes
**Provenance:** Synthesized from AnySearch research conducted 2026-07-31. All citations are peer-reviewed unless marked `[preprint]` or `[vendor]`.

---

## The Framing

"RF as compute" is a **synthesis label**, not an established field name. The work is distributed across four historically separate communities that don't share vocabulary, tooling, or a developer surface:

1. Computational metamaterials (physics)
2. Over-the-air computation / AirComp (information theory)
3. Microwave photonics (electrical engineering)
4. Spin-torque neuromorphic RF (condensed matter)

The unifying claim across all four: **the waveform is the operand.** Interference isn't noise to cancel — it's the multiply-accumulate operation. The medium is the math.

The opportunity is the gap: there is no unified developer surface. The four lineages publish in *Nature*, *Science*, *IEEE Trans. Inf. Theory* — and nowhere a builder can land. Whoever builds the bridge — the SDK, the simulator, the "hello world" — defines the frame for the field.

---

## The Four Lineages

### Lineage 1 — Computational Metamaterials

**One-line:** A passive or reconfigurable material performs a mathematical operation on a wavefield by scattering it.

| Role | Citation |
|---|---|
| Origin | Silva et al., "Performing mathematical operations with metamaterials," *Science* 343, 160–163 (2014). University of Pennsylvania (Engheta group). |
| SOTA | "Programmable wave-based analog computing metastructure," *Nature Communications* (2025). arXiv:2301.02850 |
| Review | Zangeneh-Nejad, Sounas, Alù & Fleury, "Analogue computing with metamaterials," *Nature Reviews Materials* 6, 207–225 (2021). 560 citations. |
| ENZ calculus | Li et al., "Performing calculus with epsilon-near-zero metamaterials," *Science Advances* 8(30), eabq6198 (2022). Tsinghua, Yue Li group. |
| Equation solver | Mohammadi Estakhri, Edwards & Engheta, "Inverse-designed metastructures that solve equations," *Science* 363, 1333–1338 (2019). Penn. |

**Key labs:** Engheta (Penn) — central figure; Alù (CUNY/ASRC); Fleury (EPFL); Yue Li (Tsinghua).

**SOTA device (the anchor paper):** 45 MHz programmable metastructure with voltage-controlled phase shifters + amplifiers as "multiplier modules." Open-loop = matrix-vector multiply (~0.001 relative error). Closed-loop with feedback = matrix inversion / equation solving (~0.005 error). Demonstrates matrix inversion (stationary), Newton's method root-finding, and Lagrangian constrained optimization (non-stationary). Authors note the module could be implemented at RF (GHz) and photonic (THz) platforms, same principle.

**SDR-mappable primitive — "metasurface kernel":**
The metastructure's multiplier module (voltage-controlled phase shifter + amplifier) maps 1:1 to an SDR's IQ modulator + variable gain stage. The metamaterial's transfer function = the impulse response programmed into the SDR's digital filter chain.

```
metasurface_kernel(operator_matrix):
    # operator_matrix → SDR transmit filter coefficients
    # wave propagation through the chain = the operation
    # received waveform = the result
```

**Open problem (deferred to community):** Scaling from 5×5 to NN-scale (1024×1024) matrices. Subwavelength resonator design at GHz frequencies. Noise resilience outside lab conditions.

---

### Lineage 2 — Over-the-Air Computation (AirComp)

**One-line:** The wireless channel itself performs the computation — "computing while communicating."

| Role | Citation |
|---|---|
| Origin | Nazer & Gastpar, "Compute-and-forward: Harnessing interference through structured codes," *IEEE Trans. Inf. Theory* 57(10), 6463–6486 (2011). Founding result. ~18 years old. |
| Survey | "Over-the-Air Computation for 6G: Foundations, Technologies, and Applications," arXiv:2210.10524 (2022) `[preprint]` |
| Survey | "A Survey on Over-the-Air Computation," arXiv:2210.11350 (2022) `[preprint]` |
| Recent theory | Eldar (MIT/Technion), Goldsmith (Stanford), Gündüz (Imperial College) |
| RIS + FL | "Empowering Over-the-Air Personalized Federated Learning via RIS," arXiv:2408.12162 (2024) `[preprint]` |

**Core claim (the user's "RF as operand" framing, stated plainly):**
AirComp treats channel interference as a *contributor* to function computation, not a disruptor. All devices share full bandwidth; no per-device demodulation. RIS (Reconfigurable Intelligent Surfaces) sculpts the channel — which means **RIS sculpts the operator.** Federated learning gradient aggregation over-the-air is a dense 2024–2025 subfield.

> "Unlike conventional multiple-access schemes considering co-channel interference as a disruptor of wireless transmission, AirComp treats the interference as a contributor to the function computation and forgoes the interference cancellation for decoding each of the data."
> — arXiv:2210.10524

**SDR-mappable primitive — "aircomp kernel":**
Distributed AirComp sum. N devices transmit simultaneously; the receiver reads the sum (or arbitrary function) from the superposed waveform. This is the most accessible "hello world" — no metamaterial hardware needed; the "channel" is free space.

```
aircomp_kernel(signals, function="sum"):
    # N SDRs transmit pre-coded signals simultaneously
    # the channel superposition IS the function evaluation
    # receiver reads f(x1, x2, ..., xN) without decoding any xi
    # no per-device demodulation — interference is the operation
```

**Open problem (deferred to community):** Channel equalization under real multipath. Security (a malicious transmitter corrupts the computation). Power alignment across distributed transmitters. Integration into 6G standardization (3GPP) — AirComp is a research candidate, not yet in 3GPP study items.

---

### Lineage 3 — Microwave Photonic Computing

**One-line:** RF signals on optical carriers, performing neural-network inference and matrix-vector multiplication.

| Role | Citation |
|---|---|
| Foundational review | Marpaung, Yao & Capmany, "Integrated microwave photonics," *Nature Photonics* 13, 80–90 (2019) |
| SOTA (NN) | Guan & Yao, "A Microwave Photonic Neural Network in the Frequency Synthetic Dimension," *J. Lightwave Technol.* 43, 9934–9940 (2025). 55×10⁶ MAC/s, 95.93% prediction accuracy (reported). |
| SOTA (tensor core) | "Photonic tensor core in thin-film lithium niobate," *Nature Communications* (2024). 120 GOPS, 60 GHz weight-update speed. In-situ training demonstrated. |

**Key name:** Jianping Yao (Ottawa) — the through-line; co-author of the 2019 review and the 2025 NN paper.

**Position in this map:** This lineage is the most commercially adjacent — but the carrier is **optical, not pure-RF**. RF drives the modulators that set weights; optical interference does the MAC. The GitHub preview should surface it as a **sibling**, not a core primitive. Lightmatter ($4.4B valuation, Passage M1000 interposer announced March 2025, ramping late 2025) is the commercial frontier here, and it validates wave-based compute — but not pure-RF compute.

**SDR-mappable primitive — "microwave photonic kernel" (simulation only):**
The optical domain doesn't map to SDR directly. However, the RF-side modulator chain (DP-MZM, carrier-suppressed single-sideband modulation) is SDR-representable. A microwave photonic kernel can be simulated in RF-only mode — losing the optical bandwidth advantage but preserving the signal-processing structure for education.

```
microwave_photonic_kernel(matrix, vector):
    # SIMULATION ONLY — no optical hardware
    # model: RF tones → modulator sidebands → coupling = MVM
    # the coupling coefficients between carriers and sidebands = the weights
    # educational: shows the structure without the optical bandwidth
```

**Open problem (deferred to community):** On-chip integration. Optical loss. In-situ training at scale. Commercial fabrication (Lightmatter is pursuing this; not our lane).

---

### Lineage 4 — Spin-Torque Neuromorphic RF

**One-line:** The "neuron" is a microwave-frequency spin-torque nano-oscillator. Computation happens *in* the microwave dynamics.

| Role | Citation |
|---|---|
| Canonical | Torrejon et al., "Neuromorphic computing with nanoscale spintronic oscillators," *Nature* 547, 428–431 (2017). 99.6% spoken-digit recognition. CNRS-Thales / Paris-Saclay / NIST. |
| Review | Grollier, Querlioz, Camsari et al., "Neuromorphic Spintronics," *Nature Electronics* 3(7) (2020) |
| Recent | Marrows et al., "Neuromorphic computing with spintronics," *Nature* (2024). 99 citations. Confirms reservoirs "read via microwave absorption." |

**Key labs:** Grollier (CNRS-Thales) — central; Querlioz (Paris-Saclay); Camsari (Purdue); Stiles (NIST).

**SDR-mappable primitive — "reservoir kernel" (approximate simulation):**
The spin-torque oscillator's dynamics (nonlinear microwave oscillation with relaxation) can be approximated as an RF resonator with a nonlinear response curve. Drive an RF resonator with a pre-coded signal, read the state via reflected power, use the nonlinear mixing as a feature transform. This is the most speculative SDR mapping — the physics (magnetic tunnel junctions) is hard to emulate in software-defined hardware.

```
reservoir_kernel(input_signal):
    # APPROXIMATE — emulates spin-torque oscillator dynamics in RF
    # model: nonlinear RF resonator with relaxation oscillation
    # drive with input → nonlinear mixing → read reflected power
    # the nonlinear response = the "neuron" activation
    # educational: shows reservoir computing structure
    # NOT a replacement for magnetic tunnel junction hardware
```

**Open problem (deferred to community):** Hardware fabrication (magnetic tunnel junctions are not SDR-reproducible). Scaling beyond spoken-digit-class tasks. Device-to-device variability. Readout integration.

---

## The SDR Bridge — Why Software-Defined Radio Is the Developer Surface

The four lineages share a hidden common substrate: **they all manipulate RF waveforms to perform computation.** The developer surface that connects them is Software-Defined Radio (SDR), because:

| Lineage | Metastructure / hardware element | SDR equivalent |
|---|---|---|
| Metamaterials | Voltage-controlled phase shifter + amplifier (multiplier module) | IQ modulator + variable gain stage |
| AirComp | Distributed transmitters + channel superposition | Multiple SDRs transmitting pre-coded signals |
| Microwave photonics | RF-driven optical modulator chain (DP-MZM, CS-SSB) | SDR modulator chain (RF-only simulation, no optical bandwidth) |
| Spin-torque | Microwave nano-oscillator with nonlinear dynamics | Nonlinear RF resonator (approximate simulation) |

SDR is the bridge because it's the one platform where a builder can touch RF waveforms as computational operands **without a fab, without a metamaterial, without a clean room.** Two HackRFs and a laptop is the entry point.

---

## The "Hello World" Ladder

Three reproducible experiments, escalating in cost and complexity. Each maps to a peer-reviewed result.

### Tier 1 — AirComp Sum (the Nazer/Gastpar 2007 result, made legible)

| Field | Value |
|---|---|
| **Cost** | ~$350 (2× HackRF One SDRs + 1× RTL-SDR + 10 MHz clock cable + laptop). Budget option ~$190 (1 HackRF + 1 RTL-SDR, sequential — see walkthrough). |
| **What it proves** | Two transmitters send pre-coded signals; the receiver reads their sum from the superposed waveform — without decoding either individual signal. |
| **The lesson** | Interference IS computation. The channel is the adder. |
| **Citation** | Nazer & Gastpar, *IEEE Trans. Inf. Theory* (2011) |
| **Difficulty** | Beginner — the simplest wave-compute primitive |
| **Walkthrough** | `docs/hello-world-aircomp.md` |

### Tier 2 — Wave-Domain Convolution / Filtering

| Field | Value |
|---|---|
| **Cost** | ~$200 (1× HackRF One Tx + 1× RTL-SDR Rx + laptop). Mode B (passive scatterer) adds ~$10–25. |
| **What it proves** | An RF waveform passed through a programmable filter chain performs convolution in the wave domain. The filter's impulse response = the operator. |
| **The lesson** | Linear operators are native to wave physics. Convolution is not a digital algorithm here — it's what the medium *does*. |
| **Citation** | Silva et al., *Science* (2014); Zangeneh-Nejad review, *Nature Reviews Materials* (2021) |
| **Difficulty** | Intermediate — requires filter design + calibration |
| **Walkthrough** | `docs/hello-world-convolution.md` |

### Tier 3 — Matrix Inversion via Feedback (the Nature Comms 2025 result)

| Field | Value |
|---|---|
| **Cost** | Free (Mode A: simulation) / ~$200 (Mode B: software-in-the-loop, 1 HackRF + 1 RTL-SDR) / ~$350 (Mode C: analog feedback loop, adds circulator + attenuator + splitter). |
| **What it proves** | Closed-loop RF feedback solves matrix inversion / linear equations in the wave domain. Open-loop does the multiply; the feedback loop does the inversion. The wave domain *settles* to the solution. |
| **The lesson** | Iterative algorithms (stationary fixed-point / Richardson-Jacobi-type, Newton's method, Lagrangian optimization) have wave-domain implementations. Computation is not bound to clock cycles — it's bound to settling time. |
| **Citation** | *Nature Communications* (2025), arXiv:2301.02850 |
| **Difficulty** | Advanced — Mode A (simulation) is beginner-friendly; Mode B requires calibration + channel estimation; Mode C requires analog feedback loop stabilization. |
| **Walkthrough** | `docs/hello-world-matrix-inversion.md` |

---

## What We Are NOT Building (graceful deferral to the community)

This surface is **research and education**, not a product. The following problems are real, important, and explicitly left to the community:

| Problem | Why it's deferred | Who owns it |
|---|---|---|
| **Scaling to deep-learning-size matrices** | SOTA is 5×5. NN layers are 1024×1024. This is a hardware physics problem, not a software problem. | Metamaterials physics community (Engheta, Alù, Fleury, Yue Li) |
| **Phone-form-factor integration** | 45 MHz ≈ 6.7m wavelength. At 5 GHz, subwavelength resonator design at phone scale is unsolved. | Applied metamaterials + antenna engineering |
| **Noise resilience in hostile EM environments** | Analog compute is precision-fragile. Lab results (~0.001 error) won't hold in a pocket next to a battery and display. | RF engineering + analog circuit design |
| **6G standards integration** | AirComp is a 6G research candidate, not yet in 3GPP standardization work items. Standards work is a multi-year institutional process. | 3GPP, wireless standards bodies |
| **Commercial fabrication** | Lightmatter ($4.4B) is the commercial wave-compute frontier, and it's optical (Passage M1000 announced March 2025, ramping late 2025). Pure-RF commercial compute is not this project's ambition. | Industry (Lightmatter, Lightelligence, Salience Labs) |

**We build the map, the vocabulary, the reproducible "hello world." The community builds the future.**

---

## Public-Facing Naming Note

If this surface becomes a public GitHub repository, the name should be clean and descriptive (not an internal project codename). Suggested public names:

| Name | Character |
|---|---|
| `rf-compute` | Clean, descriptive, field-naming. Default recommendation. |
| `wavecompute` | Broader, includes optical siblings. |
| `rf-as-compute` | Explicit framing, SEO-friendly for the gap. |

Internal provenance is preserved in this document; the public surface carries its own clean identity.

---

## Field Maturity at a Glance

| Lineage | Theory age | Peer-reviewed hardware | Commercial | SDR bridge maturity |
|---|---|---|---|---|
| AirComp / RIS | ~18 yrs (2007) | RIS hardware exists | 6G research candidate, not deployed | **Highest** — pure SDR, no custom hardware |
| Computational metamaterials | ~12 yrs (2014) | Yes (2025 RF metastructure) | Lab-stage | **High** — multiplier module maps to SDR IQ chain |
| Microwave photonics | ~16 yrs (2009) | Yes (TFLN tensor core) | Adjacent (Lightmatter, optical) | **Low** — optical domain doesn't map to SDR directly |
| Spin-torque neuromorphic | ~9 yrs (2017) | Yes (nano-oscillator arrays) | Lab-stage | **Speculative** — physics hard to emulate in SDR |

---

## Provenance

- **Research method:** AnySearch MCP, 2026-07-31
- **Citation verification:** All citations checked as peer-reviewed unless marked `[preprint]` or `[vendor]`
- **Honesty flags:** Lightmatter performance numbers are vendor-sourced. AirComp surveys are preprints. Spin-torque SDR mapping is approximate, not a hardware equivalent.