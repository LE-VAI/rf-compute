# Annotated Bibliography

## The reading list for wave-domain computation

Every paper cited in the hello-world walkthroughs, organized by lineage, with one paragraph each on **what it claims** and **why it matters for this surface**. Read top-to-bottom within a lineage; the reading order at the end gives a cross-lineage path through the field.

Citations are peer-reviewed unless marked `[preprint]` or `[vendor]`. Where a paper sits in the lineage, the corresponding walkthrough names it.

---

## Lineage 1 — Computational Metamaterials

> The "wave does math" lineage. A material performs a mathematical operation on a wavefield by scattering it. Origin: University of Pennsylvania (Engheta group), 2014. Central figure: Nader Engheta (Penn).

### Silva et al. 2014 — *Science* — The origin paper

**Citation:** Silva, A., Monticone, F., Alù, A. et al. "Performing mathematical operations with metamaterials." *Science* 343, 160–163 (2014). [DOI: 10.1126/science.1242818](https://doi.org/10.1126/science.1242818)

**What it claims:** A passive metamaterial can perform a mathematical operation — differentiation, integration, convolution, correlation — on an incident wavefield by engineered scattering. The material's transfer function *is* the operator; no electronics, no algorithms.

**Why it matters for this surface:** This is the founding paper of computational metamaterials. Tier 2 of the hello-world ladder (`docs/hello-world-convolution.md`) demonstrates the same principle — convolution as a wave-native operation — using an SDR FIR filter as a programmable stand-in for the metamaterial. The Silva paper is the "this is possible" claim; the Tier 2 walkthrough is the "you can touch it for $200" version.

---

### Mohammadi Estakhri, Edwards & Engheta 2019 — *Science* — Equation-solving metastructures

**Citation:** Mohammadi Estakhri, N., Edwards, B. & Engheta, N. "Inverse-designed metastructures that solve equations." *Science* 363, 1333–1338 (2019). [DOI: 10.1126/science.aaw2498](https://doi.org/10.1126/science.aaw2498)

**What it claims:** A metastructure can solve a system of linear equations `Ax = b` by scattering — the solution `x` is encoded in the outgoing wave. The geometry is inverse-designed to realize the operator `A`.

**Why it matters for this surface:** This is the direct intellectual ancestor of the 2025 Nature Communications metastructure (the Tier 3 anchor paper). Engheta's group showed that solving equations — not just applying operators — is possible in the wave domain. Tier 3 of the hello-world ladder demonstrates the same mathematical principle (matrix inversion via feedback) at three levels of fidelity; this paper is the "you can solve equations in the wave domain" claim that Tier 3 makes bench-accessible.

---

### Zangeneh-Nejad, Sounas, Alù & Fleury 2021 — *Nature Reviews Materials* — The canonical review

**Citation:** Zangeneh-Nejad, F., Sounas, D.L., Alù, A. & Fleury, R. "Analogue computing with metamaterials." *Nature Reviews Materials* 6, 207–225 (2021). [DOI: 10.1038/s41578-020-00243-2](https://doi.org/10.1038/s41578-020-00243-2)

**What it claims:** A comprehensive review of wave-based analog computing — the basic principles, recent advances, and future directions. Establishes "wave-based analog computing" as the umbrella term. Covers computational metamaterials and metasurfaces, including subwavelength resonant scatterer design.

**Why it matters for this surface:** This is the canonical reference for anyone who wants the full landscape, not just the SOTA. The 560-citation count signals it's the field's shared vocabulary. If an educator assigns one reading before the hello-world ladder, it should be this — it gives the theoretical context the walkthroughs assume. The "subwavelength" discussion in the review is the link to the field map's "phone-form-factor" deferral.

---

### Li et al. 2022 — *Science Advances* — ENZ calculus

**Citation:** Li, Y. et al. "Performing calculus with epsilon-near-zero metamaterials." *Science Advances* 8(30), eabq6198 (2022). [DOI: 10.1126/sciadv.abq6198](https://doi.org/10.1126/sciadv.abq6198)

**What it claims:** An epsilon-near-zero (ENZ) metamaterial performs differentiation and integration — calculus operations — on analog signals at subwavelength scale, by generating the desired dispersions of the ENZ metamaterials with photonic doping.

**Why it matters for this surface:** This is the Tsinghua / Yue Li group's contribution to the lineage — the ENZ near-zero-index line that the field map names. The Tier 2 walkthrough's "differencer" operator (the `[1, −1]` FIR) is the digital stand-in for what this paper does in a material. The ENZ result is the bridge from "wave-native convolution" to "wave-native calculus" — and subwavelength, on-chip compatible, which is why it's the path to the phone-form-factor frontier the field defers.

---

### Nature Communications 2025 — The SOTA anchor paper

**Citation:** "Programmable wave-based analog computing metastructure." *Nature Communications* (2025). [arXiv:2301.02850](https://arxiv.org/abs/2301.02850)

**What it claims:** A reconfigurable RF metastructure at 45 MHz performs matrix inversion (stationary problem), Newton's method root-finding, and Lagrangian constrained optimization (non-stationary problems) in the wave domain. Open-loop = matrix-vector multiply (~0.001 relative error); closed-loop with feedback = matrix inversion / equation solving (~0.005 error). The device uses voltage-controlled phase shifters + amplifiers as "multiplier modules." Authors note the module could be implemented at RF (GHz) and photonic (THz) platforms.

**Why it matters for this surface:** This is the single strongest "RF as compute" paper as of 2026. It is *physically built* (not simulation-only) and *experimentally verified* (not theoretical). Tier 3 of the hello-world ladder (`docs/hello-world-matrix-inversion.md`) demonstrates the same closed-loop inversion principle at three fidelity levels. The 5×5 matrix scale and the ~0.005 error are the frontier limits the field map defers to the community.

---

## Lineage 2 — Over-the-Air Computation (AirComp)

> The "computing while communicating" lineage. The wireless channel itself performs the computation. Origin: Nazer & Gastpar, 2007. 6G research candidate (not yet in 3GPP standardization).

### Nazer & Gastpar 2007/2011 — *IEEE Trans. Inf. Theory* — The founding result

**Citation:** Nazer, B. & Gastpar, M. "Computation over multiple-access channels," *IEEE Trans. Inf. Theory* 53(10), 3498–3516 (2007) — the founding result. Namesake extension: "Compute-and-forward: Harnessing interference through structured codes," *IEEE Trans. Inf. Theory* 57(10), 6463–6486 (2011). [DOI: 10.1109/TIT.2011.2165816](https://doi.org/10.1109/TIT.2011.2165816).

**What it claims:** Lattice-coded interference over a multiple-access channel can compute functions (sums) directly in the wave domain — the receiver recovers the function of the transmitted messages without decoding each message individually. Interference is harnessed as computation, not cancelled as noise.

**Why it matters for this surface:** This is the origin point of "RF as compute" as a formal information-theoretic concept — ~18 years old, not a new idea. Tier 1 of the hello-world ladder (`docs/hello-world-aircomp.md`) demonstrates the analog-superposition version of this result (the pedagogical core). The lattice-coded finite-field version (noise-resilient and exact) is flagged in the walkthrough as a "Tier 1.5" extension and named as a contribution target in `CONTRIBUTING.md`.

---

### arXiv:2210.10524 — AirComp for 6G survey `[preprint]`

**Citation:** Wang, J. et al. "Over-the-Air Computation for 6G: Foundations, Technologies, and Applications." arXiv:2210.10524 (2022). [arXiv: 2210.10524](https://arxiv.org/abs/2210.10524)

**What it claims:** A survey of AirComp theory, the canonical framing paper for 6G-targeted ambient RF compute. States the central AirComp claim explicitly: "AirComp treats the interference as a contributor to the function computation and forgoes the interference cancellation for decoding each of the data."

**Why it matters for this surface:** This is where the "RF as operand" framing is stated most plainly in the literature. The user's unifying label — "RF for information carriers AND computational operands" — maps directly onto this survey's thesis. Cited in the field map as the AirComp framing reference. Preprint (peer-reviewed venue unclear) — treat as authoritative but not yet peer-reviewed.

---

### arXiv:2210.11350 — AirComp survey `[preprint]`

**Citation:** "A Survey on Over-the-Air Computation." arXiv:2210.11350 (2022). [arXiv: 2210.11350](https://arxiv.org/abs/2210.11350)

**What it claims:** A survey establishing AirComp as a subfield with its own taxonomy — communication and computation are treated as a unified task when the receiver wants a function of the local information at the devices, rather than the local information itself.

**Why it matters for this surface:** Pairs with the 6G survey to confirm AirComp is an established subfield, not a fringe idea. Use this as the broader reference; use the 6G survey (above) for the 6G-specific framing. Preprint.

---

### arXiv:2408.12162 — RIS + AirComp + Federated Learning `[preprint]`

**Citation:** Shi, G. et al. "Empowering Over-the-Air Personalized Federated Learning via RIS." arXiv:2408.12162 (2024). [arXiv: 2408.12162](https://arxiv.org/abs/2408.12162)

**What it claims:** Over-the-air computation integrates analog communication with task-oriented computation for communication-efficient federated learning; RIS (Reconfigurable Intelligent Surfaces) enables personalized AirFL by mitigating data heterogeneity.

**Why it matters for this surface:** Representative of the dense 2024–2025 RIS + AirComp + Federated Learning literature. The propagation environment is treated as a reconfigurable compute substrate — RIS sculpts the channel, which means RIS sculpts the operator. This is the 6G-adjacent ambient-RF-compute frontier the field map names as the most mature angle.

---

## Lineage 3 — Microwave Photonic Computing

> The "RF-modulated light" lineage. RF signals on optical carriers performing neural-network inference. Hybrid, not pure-RF. Central figure: Jianping Yao (Ottawa). Commercially most-adjacent (via Lightmatter), but the commercial products are optical.

### Marpaung, Yao & Capmany 2019 — *Nature Photonics* — The foundational review

**Citation:** Marpaung, D., Yao, J. & Capmany, J. "Integrated microwave photonics." *Nature Photonics* 13, 80–90 (2019). [DOI: 10.1038/s41566-018-0310-5](https://doi.org/10.1038/s41566-018-0310-5)

**What it claims:** A review of microwave photonics as a discipline — the substrate (RF signals on optical carriers) from which the neural-network and tensor-core papers grow.

**Why it matters for this surface:** The foundational reference for the lineage. Yao appears again in the 2025 NN paper; he is the through-line. Use this to understand why microwave photonics is a sibling to pure-RF compute, not a synonym: the carrier is optical, the modulators are RF-driven.

---

### Guan & Yao 2025 — *J. Lightwave Technology* — The SOTA RF-photonic NN

**Citation:** Guan, Y. & Yao, J. "A Microwave Photonic Neural Network in the Frequency Synthetic Dimension Using Multi-tone Single-Sideband Modulation in a Fiber Loop." *J. Lightwave Technol.* 43, 9934–9940 (2025). [DOI: 10.1109/JLT.2025.3579197](https://doi.org/10.1109/JLT.2025.3579197)

**What it claims:** A microwave photonic neural network using a dual-parallel Mach-Zehnder modulator (DP-MZM) in an optical fiber loop. Multi-tone RF signals applied to the modulator generate sidebands; the coupling between optical carriers and modulated sidebands *is* the matrix-vector multiplication. Achieves 55×10⁶ MAC/s and 95.93% prediction accuracy on a 3-2-8 feedforward NN.

**Why it matters for this surface:** The strongest "RF photonic computing" result. The RF signals ARE the computational operands — the MVM is literally the coupling between RF-driven sidebands. Experimentally demonstrated, not theoretical. This is the lineage the field map names as "most commercially advanced adjacent" — note the modulators are RF-driven, but the bandwidth advantage is optical.

---

### Nature Communications 2024 — TFLN photonic tensor core

**Citation:** "Photonic tensor core in thin-film lithium niobate." *Nature Communications* (2024). [DOI: 10.1038/s41467-024-53261-x](https://doi.org/10.1038/s41467-024-53261-x)

**What it claims:** A fully integrated photonic tensor core — two thin-film lithium niobate (TFLN) modulators, a III-V laser, and a charge-integration photoreceiver — implementing an entire NN layer at 120 GOPS with 60 GHz weight-update speed. In-situ training (supervised + unsupervised) demonstrated on 112×112 images.

**Why it matters for this surface:** TFLN modulators are electro-optic — driven by RF/microwave signals. RF voltages set the weights; optical interference does the MAC. The 60 GHz weight-update is squarely microwave-domain. This is the commercial frontier's lab-stage precursor. Cited in the field map alongside the Guan/Yao result as the SOTA of the lineage.

---

## Lineage 4 — Spin-Torque Neuromorphic RF

> The "RF oscillator is the neuron" lineage. The "neuron" is a microwave-frequency (GHz) spin-torque nano-oscillator. Computation happens *in* the microwave dynamics. Central figure: Julie Grollier (CNRS-Thales).

### Torrejon et al. 2017 — *Nature* — The canonical neuromorphic-RF paper

**Citation:** Torrejon, J. et al. "Neuromorphic computing with nanoscale spintronic oscillators." *Nature* 547, 428–431 (2017). [DOI: 10.1038/nature23011](https://doi.org/10.1038/nature23011). PMC: [PMC5575904](https://pmc.ncbi.nlm.nih.gov/articles/PMC5575904/)

**What it claims:** Spin-torque nano-oscillators — magnetic tunnel junctions driven by DC current into sustained magnetization precession — perform neuromorphic computation via time-multiplexed reservoir computing. The oscillation is converted to voltage via tunneling magnetoresistance. Achieves 99.6% spoken-digit recognition — state-of-the-art for existing hardware and software at the time.

**Why it matters for this surface:** The canonical neuromorphic-RF paper. The "neuron" is a microwave-frequency (GHz) spin-torque nano-oscillator — the computation happens IN the microwave dynamics, with time-multiplexed reservoir computing. This is RF-as-compute where the RF oscillator is the neuron, not a transmission medium. The 99.6% result is the proof that this lineage does real ML tasks, not just toy demos.

---

### Grollier, Querlioz, Camsari et al. 2020 — *Nature Electronics* — The field-defining review

**Citation:** Grollier, J., Querlioz, D., Camsari, K. et al. "Neuromorphic Spintronics." *Nature Electronics* 3(7), 360–370 (2020). [escholarship full text](https://escholarship.org/content/qt2fc8111x/qt2fc8111x.pdf)

**What it claims:** A review positioning spin-torque oscillators (microwave devices) as a neuromorphic primitive. The field-defining review for spintronic neuromorphic compute.

**Why it matters for this surface:** Establishes the lineage's vocabulary and the key figures: Grollier (CNRS-Thales), Querlioz (Paris-Saclay), Camsari (Purdue). Use this as the entry point for the lineage; use Torrejon 2017 for the canonical result.

---

### Marrows et al. 2024 — *Nature* — The 2024 status update

**Citation:** Marrows, I. et al. "Neuromorphic computing with spintronics." *Nature* (npj/experimental), 2024. [DOI: 10.1038/s44306-024-00019-2](https://doi.org/10.1038/s44306-024-00019-2)

**What it claims:** A 2024 review confirming the spin-torque-neuromorphic line is active and explicitly microwave-coupled — reservoir systems can be "read via microwave absorption."

**Why it matters for this surface:** Confirms the lineage is not dormant. The "read via microwave absorption" framing makes the RF coupling explicit. Use this to check the 2024 state of the field after reading the 2020 review.

---

## Lineage 3 (adjacent) — Commercial frontier

> The only substantial commercial activity in wave-based compute as of 2026. Optical, not pure-RF, but uses RF-driven modulators. Vendor-sourced performance numbers — treat with standard diligence.

### Lightmatter `[vendor]`

**Citations:** Lightmatter company publications and press: [lightmatter.co](https://lightmatter.co), [Reuters 2025-04-01](https://www.reuters.com/technology/artificial-intelligence/lightmatter-releases-new-photonics-technology-ai-chips-2025-04-01/), [TechCrunch 2024-10-16](https://techcrunch.com/2024/10/16/lightmatters-400m-d-round-has-ai-hyperscalers-hyped-for-photonic-datacenters/)

**What it claims:** Lightmatter (MIT spin-out) builds photonic AI chips using Mach-Zehnder interferometers performing GEMM with light. Envise = processor; Passage M1000 = 3D photonic interposer (114 Tbps optical bandwidth, up to 1.6 Tbps per fiber, 256 fibers per chip). $4.4B valuation, $850M raised, Google/T. Rowe Price backed. Interposer announced March 2025, ramping late 2025; chiplet 2026.

**Why it matters for this surface:** The ONLY substantial commercial wave-compute player found. Validates wave-based compute commercially — but the carrier is optical, not pure-RF. The modulators that set weights are RF-driven, so Lightmatter sits in the same physics family as the microwave photonic lineage. Cited in the field map as the commercial frontier. Performance numbers are vendor-sourced.

---

## Reading order

If you're new to the field and want to read your way in, here's a cross-lineage path that builds from foundational to frontier, taking roughly 8–10 hours of focused reading.

### Phase 1 — The core insight (2 hours)

1. **Zangeneh-Nejad 2021** (review) — the umbrella vocabulary and the full landscape in one paper
2. **Silva 2014** (Science) — the origin: a material does math on a wavefield

### Phase 2 — The four lineages (4 hours)

3. **Nazer & Gastpar 2011** (IEEE TIT) — interference is computation, not noise
4. **Torrejon 2017** (Nature) — a microwave oscillator is a neuron
5. **Marpaung/Yao/Capmany 2019** (Nature Photonics) — the microwave-photonic substrate
6. **arXiv:2210.10524** (AirComp 6G survey) — the "RF as operand" framing, made plain

### Phase 3 — The frontier (2 hours)

7. **Nature Communications 2025** (arXiv:2301.02850) — the SOTA: a programmable RF metastructure solving equations
8. **Guan & Yao 2025** (JLT) — the SOTA of microwave photonic NN

### Phase 4 — The context (1 hour)

9. **Marrows 2024** (Nature) — the 2024 status of spin-torque neuromorphic
10. **Lightmatter press** (vendor) — the commercial frontier (with appropriate skepticism)

After Phase 1 you have the vocabulary. After Phase 2 you have the four-lineage map. After Phase 3 you know where the frontier is. After Phase 4 you know who's trying to commercialize it.

Then run the hello-world ladder.

---

## How to use this bibliography

- **Educators:** Assign Phase 1 + the Tier 1 walkthrough for a one-week unit on wave-domain compute. Assign Phase 3 + the Tier 3 walkthrough for a graduate seminar on the frontier.
- **Researchers entering the field:** Read Phases 1–3 in order. The hello-world ladder is optional but recommended — it will make the papers' claims physically tangible.
- **Builders:** Skip to the walkthroughs first. Come back here when a walkthrough says "maps to [paper]" and you want the depth.
- **Contributors:** See `CONTRIBUTING.md` for the open contribution targets — each maps to a gap the bibliography surfaces.

---

## Provenance

- **Research method:** AnySearch MCP, 2026-07-31
- **Verification:** All citations checked as peer-reviewed unless marked `[preprint]` or `[vendor]`. Lightmatter performance figures are vendor-sourced. AirComp surveys are preprints.
