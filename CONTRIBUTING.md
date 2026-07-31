# Contributing to rf-compute

## What this project is — and what it isn't

`rf-compute` is a **research and education surface** for wave-domain computation. It exists to make "RF as compute" legible to builders — the map, the vocabulary, and the reproducible "hello world." It is **not** a product, not a framework, and not a claim that RF will replace digital compute.

This project's measure of success is: **a builder who'd never heard of RF-as-compute can run the Tier 1 experiment in a weekend, understand the field, and know where the open frontier lives.** Contributions that advance that goal are welcome. Contributions that drift toward hype, overpromise, or "RF will replace GPUs" rhetoric are not.

If you're here because you want to advance the *physics* — scaling metastructures to NN-size matrices, building phone-form-factor devices, solving the noise-resilience problem — you are exactly who we're inviting. Read the "Open frontier" section below.

---

## The contribution bar

Every contribution must clear three bars. PRs that miss any of them will be held until they do.

### 1. Reproducibility

An experiment is not real until someone else can run it. Every hardware walkthrough contribution must include:

- [ ] **Parts list with costs** — exact SDR model, cables, antennas, passive components, with current prices
- [ ] **Software dependencies pinned** — Python version, library versions, SDR driver versions
- [ ] **Step-by-step setup** — what to plug in, what to install, what to run, in order
- [ ] **Expected output** — what the builder should see, with a quantitative check (e.g., "correlation > 0.7" or "received sum within 10% of x1+x2")
- [ ] **Troubleshooting table** — the real failure modes (carrier drift, phase cancellation, clipping, DC offset) with fixes
- [ ] **Honest scope statement** — what the experiment proves and what it approximates, stated upfront

If your contribution is a new operator, a new iterative algorithm, or a new hardware configuration, it must be reproducible by a builder with the listed parts. "Works on my machine" is not reproducible.

### 2. Provenance

Every claim must trace to a source. No unsourced assertions.

- [ ] **Peer-reviewed citation** for every physics or information-theory claim (journal, DOI, year)
- [ ] **`[preprint]`** marker for arXiv-only results
- [ ] **`[vendor]`** marker for company-sourced performance numbers, with a note to treat with standard diligence
- [ ] **"Honest scope"** paragraph in every walkthrough stating what the demo proves vs. what the cited paper does

If you're making a claim that isn't in the literature, label it as **[experimental]** or **[hypothesis]** and explain what would test it. The field is full of simulation-only metasurface results that don't reproduce; this surface will not add to that pile.

### 3. Honesty about limits

Every contribution must name its own limits. The field's biggest risk is overpromising.

- [ ] **Precision floor stated** — analog compute is noise-limited; say what precision your demo achieves (e.g., "converges to ~3 decimal places with careful tuning")
- [ ] **Scale stated** — what matrix size, what bandwidth, what device count? The SOTA is 5×5; don't imply 1024×1024.
- [ ] **What this does NOT prove** — one paragraph per walkthrough explicitly naming the gap between the demo and useful compute
- [ ] **No "RF will replace GPUs" rhetoric** — this surface is research and education, not a pitch deck

The hello-world walkthroughs model this standard. Read them before contributing.

---

## What we're looking for

### High-priority contributions (the open frontier)

These are the problems the field map explicitly defers to the community. They are hard. They are where the field's future is being built. If you have the physics, the hardware, or the theory for any of these, we want to host the bridge that makes your work legible to builders.

| Contribution target | What it is | Who owns it |
|---|---|---|
| **Scaling to NN-size matrices** | SOTA is 5×5. The path to 1024×1024 is a hardware physics problem — subwavelength resonator design, frequency-multiplexed parallel computation, or photonic-bandgap structures. | Metamaterials physics community (Engheta, Alù, Fleury, Yue Li lines) |
| **Phone-form-factor integration** | 45 MHz ≈ 6.7m wavelength. Subwavelength resonator design at 5 GHz / phone scale is unsolved. ENZ (epsilon-near-zero) materials are a candidate path. | Applied metamaterials + antenna engineering |
| **Noise resilience in hostile EM environments** | Analog compute is precision-fragile. Lattice-coded AirComp (the full Nazer/Gastpar result) is the information-theoretic answer. Practical noise-resilient analog compute at scale is open. | RF engineering + analog circuit design + information theory |
| **6G standards integration** | AirComp is a 6G research candidate, not yet in 3GPP standardization work items. 3GPP integration is a multi-year institutional process. | 3GPP, wireless standards bodies |
| **Tier 1.5: Lattice-coded AirComp** | The full Nazer/Gastpar 2007/2011 result — nested lattice codes computing finite-field functions over the channel — is a software implementation project. The Tier 1 walkthrough demonstrates the analog superposition version; the lattice-coded version is the noise-resilient exact version. | Information theory + software engineering |
| **Commercial fabrication** | Pure-RF commercial compute does not exist. Lightmatter is pursuing the optical frontier. | Industry (Lightmatter, Lightelligence, Salience Labs, Ayar Labs) |

### Medium-priority contributions

- **New operators for Tier 2** — beyond boxcar / differencer / matched / Hilbert. Each new operator (with a peer-reviewed citation) extends the library.
- **New iterative algorithms for Tier 3** — beyond Richardson. Newton's method and Lagrangian optimization (the Nature Comms 2025 non-stationary results) are natural extensions.
- **Cross-lineage bridges** — explicit mappings between the four lineages. If you can show that a spin-torque reservoir (Lineage 4) is mathematically equivalent to an AirComp sum (Lineage 2) under some limit, that's a contribution.
- **Simulation-only demos for expensive hardware** — if you can't afford the circulator + attenuator for Tier 3 Mode C, a well-documented simulation that lets builders *see* the analog loop settle is valuable.
- **Educator materials** — problem sets, lecture notes, lab guides tied to the walkthroughs. If you teach this material, your scaffolding helps the next builder.

### What we will not accept

- **Unsourced hype** — "RF compute will revolutionize AI" without a citation and a precision floor
- **Simulation-only results presented as hardware results** — label your fidelity honestly
- **Vendor performance numbers without the `[vendor]` marker** — commercial claims need skepticism
- **Walkthroughs that don't include a troubleshooting table** — if you haven't failed at your own demo, you haven't built it
- **Deletion or softening of the "what this does NOT prove" sections** — the honesty is the point

---

## How to contribute

### For a new walkthrough or experiment

1. **Read the existing walkthroughs** (`docs/hello-world-*.md`) and match their structure: honest scope, hardware, math, code, expected output, troubleshooting, what it proves, going deeper, safety, provenance.
2. **Open an issue first** describing the experiment and the paper it maps to. We'll discuss scope before you write 2000 words.
3. **Reproduce it yourself.** Document the real failure modes you hit. If you didn't hit any, you didn't push it.
4. **PR with the reproducibility + provenance + honesty checklists filled.**

### For a bibliography addition

1. **Citation must be peer-reviewed** (or clearly marked `[preprint]` / `[vendor]`).
2. **Add to the right lineage** in `docs/bibliography.md`, with the one-paragraph "what it claims / why it matters" format.
3. **If the citation changes the field map** (e.g., a new SOTA), update `docs/field-map.md` too.

### For a code contribution to the SDR kernel abstraction

1. **The kernel abstraction is a future deliverable** — not yet scaffolded. Open an issue proposing the abstraction before contributing code.
2. **The abstraction must map to all four lineages**, not just one. A kernel that only covers AirComp is not the bridge; it's a re-creation of AirComp tooling.

### For a correction

1. **Open a PR or issue** with the correction and the source.
2. **Corrections to costs, hardware counts, or fidelity claims are high-priority** — the walkthroughs must be accurate. We caught the 2-SDR vs 3-SDR issue in Tier 1 during review; we want to catch every such issue.

---

## Community standards

- **Be precise about fidelity.** "Simulation," "software-in-the-loop," and "analog hardware" are different things. Use the right word.
- **Cite the paper you're standing on.** "The wave domain does convolution" is a claim; "Silva et al. 2014 showed a metamaterial performs convolution" is a cited claim.
- **Respect the deferral.** The field map names what this project does not build. Don't quietly re-include a deferred ambition in a walkthrough.
- **No marketing language.** This is a research surface, not a product. "Revolutionary," "game-changing," "next-generation" — none of these belong here.
- **Credit the physics.** This surface did not invent wave-domain compute. The four lineages did. Cite them.

---

## The vision this surface serves

Wave-domain computation is a field where the physics is settled, the engineering is hard, and the developer surface is missing. The four lineages — computational metamaterials, AirComp, microwave photonics, spin-torque neuromorphic — publish in top venues but share no vocabulary, no tooling, no "hello world." This project builds the bridge: a common developer surface that makes the field legible without overpromising what RF can do.

The bet is that a field with a shared vocabulary and a reproducible entry point attracts the contributors who can close the open frontier — scaling, phone-form-factor, noise resilience, standards integration. We do not close that frontier here. We make it possible for someone else to.

If you're reading this, you're the someone.

---

## License

Contributions are licensed under MIT (see `LICENSE`). By contributing, you agree your contributions are licensed accordingly.

---

## Provenance

- **Project:** `rf-compute` — a research and education surface for wave-domain computation
- **Research foundation:** Multi-source academic search across *Science*, *Nature*, *Nature Photonics*, *IEEE Trans. Inf. Theory*, *Journal of Lightwave Technology*, and arXiv (2026-07-31)
- **Standard:** Every contribution must clear the reproducibility, provenance, and honesty bars described above