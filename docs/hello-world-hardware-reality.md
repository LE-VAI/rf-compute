# Hello World: Hardware Reality

## Tier 4-h — the same aggregation through a *real* receiver

Tier 4 establishes that over-the-air computation works when devices arrive with
a phase and a timing offset — as long as a pilot lets the receiver measure the
aggregate channel and equalize it. That result holds on an **idealized** model:
the offsets are drawn once per round and held **constant** across the whole
block, the receiver is floating-point, and every sample window is captured.

A $2–25 ESP32-class board does none of those things. This tier inserts the
impairments the idealized model has no basis function for — a carrier offset
that **rotates**, a timing offset that **drifts**, an uncalibrated quantization
and image path, and a capture that arrives in **bursts** — and asks whether the
equalizer survives.

**It does not.** And the thing that breaks it is not the impairment anyone
expects.

---

## Honest scope

This is a **receiver-side and channel-side model**. It does not transmit, it
does not measure hardware, and it is not a hardware driver.

Three limits stated plainly:

1. **The impairment presets are not measurements.** `PHONE_GRADE`,
   `DEV_BOARD_CLOCKED` and `DEV_BOARD_UNCALIBRATED` are *stated assumptions*
   chosen to bracket the plausible range. Every number in them should be
   replaced by a bench measurement. They exist so the experiment has a subject,
   not so it has an answer.
2. **The model is toy-scaled**, like the rest of this package: one sample per
   symbol, real-valued gradient symbols, no pulse shaping, no whitened matched
   filter. The literature's stronger receivers (Shao, Gündüz & Liew 2022) would
   do better; this tier deliberately tests the *one-screen* receiver Tier 4
   actually ships.
3. **It answers a question about a model, not about a board.** The result is a
   specification — a residual-phase budget a sync stage must hit — not a claim
   that any particular board achieves or misses it.

What it *does* establish: the failure mode is **structural and predictable**,
it has a **single dominant cause**, and that cause is **not** the one the
receiver-chain discussion usually worries about.

---

## The impairments, and where they enter

The channel sums signals, not impairments, so **where** an impairment is applied
matters:

```
   per device, BEFORE the sum        once, AFTER the sum
   ─────────────────────────        ────────────────────
   CFO ramp    θ_k(j) = φ_k + c_k·j  IQ image     conj spur at −X dB
   phase noise Wiener walk on top   AGC          unknown scale on the block
   SFO drift   τ_k(j) = τ_k + s·j/N ADC          quantization at 10 bits
   residual φ_k                     duty cycle   windows never captured
```

The per-device terms cannot be commuted through the sum — that is precisely why
they are the ones that hurt. The receiver-side terms act on an already-summed
signal, and (as the numbers below show) mostly do not.

---

## What you should see

Run `python examples/tier4h_hardware_reality.py`.

### 1. The survival table

Median NMSE at a fixed point, severe misalignment (phase ~ U(−π, π),
timing ≤ 0.5 symbol), 120 channel draws. **1.0 means the estimate carries no
more information than returning zero.**

| hardware | aligned | naive | equalized | tracked | coverage |
|---|---|---|---|---|---|
| ideal (Tier 4 model) | 0.0007 | 0.001 | **0.418** | 0.418 | 1.00 |
| phone grade | 0.0007 | 1.061 | 0.571 | 0.674 | 1.00 |
| dev board, ota-synced | 0.0007 | 1.041 | 1.601 | 2.441 | 0.90 |
| dev board, uncalibrated | 0.0007 | 1.000 | **17.521** | 23.142 | 0.60 |

Read the first row as the published Tier 4 claim and the last row as what an
unsynchronised board does to it: **0.42 → 17.5**.

Note also that the **naive** receiver is *stable* across rows (≈1.0 everywhere).
That is not resilience — it is the same failure Tier 4 already documents (with
no phase sync its expected estimate is zero), and the impairment layer adds
nothing to it.

### 2. Attribution: which impairment did it

One impairment enabled at a time, from a clean receiver:

| impairment enabled alone | NMSE | vs clean |
|---|---|---|
| none (clean float, full duty) | 0.439 | 1.0 |
| **CFO ramp, 0.35 rad/symbol** | **7.178** | **16.3×** |
| phase noise, 0.02 rad/symbol | 0.446 | 1.0× |
| SFO drift, 0.25 sym/block | 0.507 | 1.2× |
| IQ imbalance, −30 dB | 0.444 | 1.0× |
| ADC 10-bit | 0.440 | 1.0× |
| AGC error, 1.0 dB | 0.439 | 1.0× |
| **burst duty cycle, 0.6** | **4.199** | **9.6×** |

This is the load-bearing table. The oscillator is the whole story; the
receiver-chain items that usually dominate an SDR quality discussion —
quantization depth, image rejection, gain calibration — are **indistinguishable
from a clean float receiver** at these power levels.

### 3. The ceiling is an accumulated-phase budget

| CFO rad/sym | phase over block | NMSE | verdict |
|---|---|---|---|
| 0.000 | 0.00 rad | 0.439 | holds |
| 0.010 | 0.20 rad | 0.455 | holds |
| 0.020 | 0.40 rad | 0.529 | holds |
| 0.040 | 0.80 rad | 0.903 | degraded |
| 0.080 | 1.60 rad | 1.888 | FAILS |
| 0.150 | 3.00 rad | 3.476 | FAILS |
| 0.350 | 7.00 rad | 7.178 | FAILS |

**The equalizer holds while the phase accumulated across the block stays under
roughly half a radian.** Above that it fails, and re-piloting does not save it:
at 0.35 rad/symbol the coherence time is about 1.4 symbols, so a midamble is
stale before its own data segment begins.

---

## The fix: a tracking receiver, and where it stops

The ceiling above is a property of the **block-fit receiver**, not of the
channel — and the way to show that is to build the receiver a real burst modem
would use. `receiver='tracking'` is a **second-order decision-directed
phase-locked loop**: it acquires on the known midamble, then tracks the
aggregate phase symbol by symbol.

**Why second order.** A residual carrier offset is a constant *frequency* error,
not a constant phase error. A first-order loop (phase accumulator only) has a
non-zero steady-state phase error under a frequency error — it lags for ever. The
integral term estimates the frequency itself, so a constant rotation is tracked
with **zero steady-state error**. That is the specific machinery the block fit
lacks.

The loop removes the **rotation**; a two-tap deconvolution on the de-rotated
stream then removes the **ISI**. Splitting the job that way is what a real
receiver does, and it matters: tracking phase alone leaves the ISI term
untouched, which is a mistake worth naming because an early version of this
receiver made it and failed *even with no carrier offset at all*.

### What it rescues

| common CFO | accumulated | equalized | **tracking** |
|---|---|---|---|
| 0.00 | 0.00 rad | 0.446 | 0.579 |
| 0.05 | 1.00 rad | 0.812 | **0.554** |
| 0.10 | 2.00 rad | 2.862 | **0.545** |
| 0.20 | 4.00 rad | 8.558 | **0.624** |
| 0.35 | 7.00 rad | 22.682 | **0.632** |
| 0.50 | 10.00 rad | 48.631 | 1.016 |

**A properly tuned tracking loop clears 7 radians of accumulation — 14× past
the block fit's half-radian wall.** The half-radian budget is therefore a
property of the *toy receiver*, not of the channel. That is the answer to the
question this tier raised.

**Loop gain is a real design parameter, not a formality.** The same loop with a
conservatively narrow gain (`kp=0.25, ki=0.01`) *also* fails at 0.35 rad/symbol.
"Add a tracking loop" is only true of a properly tuned loop; a narrow one is the
same trap in a different costume. The default here is `kp=0.8, ki=0.2`.

### What it still cannot do — the structural limit

A single-phase tracker follows **one** rotation. Independent per-device offsets
give every device its own, so the tracked phase is a compromise rather than a
fix:

| scenario | equalized | tracking |
|---|---|---|
| no CFO | 0.439 | 0.576 |
| **common** 0.10 (one rotation) | 2.926 | **0.545** |
| **independent** 0.05 (19 rotations) | 1.294 | 1.215 |
| **independent** 0.35 | 7.178 | 4.677 |

### Receiver or channel? A genie bound

How much of that is the receiver, and how much the channel? `receiver='genie'`
answers it. The genie is **not realizable**: it is handed the true combined
channel, both aggregate taps, at every symbol, and solves the time-varying
deconvolution. That bounds every receiver that models the sum's channel, however
it estimates it.

| scenario | tracking | genie |
|---|---|---|
| no CFO | 0.576 | 0.449 |
| **common** 0.35 | 0.640 | **0.441** |
| **independent** 0.35 | 4.677 | **0.778** |

Under independent offsets the genie stays useful (0.78, below the 1.0 line) and
far below the tracker's 4.68: **most of the gap is the receiver.** Not all of it.
A common rotation costs a genie nothing, but a spread still raises its error from
0.45 to 0.78, because device-to-device differences are smeared by phases that now
move. **Mostly the receiver; partly the channel.** The genie figures are medians
over 120 draws; across 120 to 400 draws the independent case stays between 0.78
and 0.86.

> **Correction (2026-10-04).** An earlier version of this section cited the
> coherence of the superposition, `|Σ_k e^{iθ_k}|/K`, as 0.2123 clean against
> 0.1994 with independent offsets, "a 6% loss", and concluded that the limit was
> architectural rather than physical. That measurement could not support the
> conclusion. With start phases uniform on (−π, π), the magnitude sits near
> √(π/4K) ≈ 0.198 for *any* phase trajectory, including phases re-drawn every
> symbol, which leave nothing to track. The 6% gap was sampling noise over 40
> draws, and the quoted figures did not reproduce from the test that cited them.
> The genie bound above replaces it.

Which sharpens the Tier 4-h conclusion rather than softening it: **the sync
target is zero residual offset, not merely a small spread.** A common rotation is
a solved problem — a tuned loop tracks it. A *spread* is not, and that is what
"unsynchronised nodes" actually means.

---

## The counterintuitive result

The obvious reading of the survival table is "the boards need to be
synchronised." That is right, but the naive version of it is **wrong in a way
worth stating**, because it changes what the sync stage must do:

| scenario | equalizer NMSE |
|---|---|
| no CFO at all | 0.439 |
| CFO **independent** per device (no sync) | 7.178 |
| CFO **common** to all devices (shared clock) | **25.072** |

A **common** carrier offset is *worse* than independent ones. Independent
offsets are random and partially average out across devices; a shared offset
makes every device's error rotate the same way, so the errors **add
coherently**. A clock that agrees on the wrong frequency is worse than clocks
that disagree.

**The sync target is zero residual offset, not a common one.**

---

## What this proves

- The pilot-aided equalizer Tier 4 ships **does not survive** a real
  ESP32-class front end, and the reason is the oscillator.
- The failure is **predictable and bounded**: an accumulated-phase budget of
  roughly half a radian per block **for the block-fit receiver**.
- **That budget is a receiver property, not a channel property.** A properly
  tuned second-order tracking loop clears 7 radians of accumulation — 14× the
  block fit's wall — and the loop gain has to be right, not merely present.
- What no single-phase receiver fixes is a **spread** of independent offsets.
  A genie handed the true combined channel shows that most of that failure is
  the receiver (genie 0.78 against the tracker's 4.68), and some of it is the
  channel (the genie's own error rises from 0.45 to 0.78).
- The receiver-chain concerns (ADC depth, IQ image, AGC) are **not** the binding
  constraint. Effort spent there is effort not spent on the LO.

## What this does NOT prove

- That any specific board misses any particular budget. The presets are
  assumptions. **The two most valuable next measurements are the residual CFO of
  real synchronised hardware, and how much of it is common versus spread** —
  those two numbers decide whether the bridge needs a better oscillator, a tuned
  tracking loop, or a multi-phase receiver.
- That a real receiver cannot do better than a *single* tracking loop. A
  multi-phase or per-device estimator is the natural next rung and is not built
  here. The genie says how much is on the table (from 4.68 down to about 0.78),
  not that a realizable receiver reaches it.
- That burst capture is fatal. It costs ~10× on its own and compounds the CFO
  failure, but it is second-order to the oscillator.

---

## The engineering consequence

The 34 tests in `tests/test_hardware.py` pin these claims as **orderings and
ratios**, not magic constants, so a change that breaks the physics fails while a
change that merely reshuffles random draws does not. Six are worth knowing by
name:

- `test_cfo_is_the_dominant_impairment` — CFO over 5× clean; ADC, IQ and AGC
  within 25% of clean.
- `test_common_clock_is_not_a_free_rescue` — shared offsets are worse than
  independent ones for the *block fit*.
- `test_equalizer_holds_under_half_a_radian_accumulated` — the block-fit budget,
  crossed from both sides.
- `test_tracking_rescues_a_common_rotation` — the tuned loop clears what the
  block fit cannot (25.1 → 0.64 at 7 rad accumulated).
- `test_loop_gain_is_a_real_parameter_not_a_formality` — a narrow loop fails
  where a tuned one holds.
- `test_tracking_cannot_rescue_independent_offsets` — the structural limit, with
  the two `test_genie_*` tests pinning how it splits: mostly the receiver
  (genie under 1.0, tracker over 3× the genie), partly the channel (a spread
  raises even the genie's error by over 30%).

## Going deeper

- **Multi-phase estimation.** The independent-offset result is a bound on a
  *single*-phase tracker. The genie bound says most of that gap is recoverable in
  principle, so a receiver that models several phase trajectories, or estimates
  the combined channel symbol by symbol, is the natural next rung and is
  genuinely open here.
- **The stronger receivers.** Shao, Gündüz & Liew 2022 (`hku-icl/MisAlignedOAC`)
  oversample with a whitened matched filter and decode with aligned-sample or
  sum-product ML estimators. Those would establish whether the residual gap at
  high CFO is a timing problem, a loop-bandwidth problem, or something else.
- **The measurement.** The bridge work this tier exists to specify: capture a
  real synchronised pair, measure residual CFO, and split it into common and
  spread components. Everything above is a model until those numbers exist.

## Safety and legality

This tier **transmits nothing** and drives no hardware. It is a numerical model
of a receiver. The capture path it describes (raw I/Q from an ESP32-class part)
is receive-only in all released implementations; transmit on this hardware class
is a separate question with licensing and spurious-emission implications, and is
out of scope here.

## Provenance

- **Disclosure and capture constraints:** ESP-SDR, espargos.net/espsdr/
  (2026-09-30) — 10-bit signed I/Q word layout, 2.2–2.7 GHz receive-only tuning,
  burst duty cycle. Follow-on work: `ESPARGOS/esp-sdr`, `h0m3us3r/eSpDR`,
  `jochenhammes/esp-sdr-trx`, `ESPARGOS/SoapyESPSDR`, `jonathanmuller/esp-ppb`.
- **Misalignment model:** Shao, Gündüz & Liew, IEEE TWC 21(6) 3951–3964 (2022) —
  the model Tier 4 already cites; this tier extends it from constant offsets to
  drifting ones.
- **Impairment models:** standard SDR practice — CFO and phase noise as a
  two-part process (ramp plus Wiener walk), SFO as symbol-rate drift, IQ
  imbalance as a complex image gain, ADC as uniform rounding. Periodic in-block
  midambles as the remedy are the standard cellular construction (LTE/NR DMRS).
- **Code:** `rf_compute/hardware.py`, `examples/tier4h_hardware_reality.py`,
  `tests/test_hardware.py`.
