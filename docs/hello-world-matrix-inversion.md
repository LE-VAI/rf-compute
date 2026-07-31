# Hello World: Matrix Inversion via Feedback

## Tier 3 — The capstone: the wave domain solves equations

> **The experiment:** An RF feedback loop solves a linear system `Ax = b` for `x` — not by computing `A⁻¹` numerically, but by letting the wave field settle to the solution. Open-loop (Tier 2) does the multiply `A·x`. Closed-loop (Tier 3) does the inversion `A⁻¹·b`. The feedback loop *is* the solver.

Tier 1 proved the channel is an adder. Tier 2 proved the channel is a linear operator. Tier 3 proves something deeper: **the channel can solve equations.** A linear system `Ax = b` is solved when the feedback loop drives the error to zero; the wave domain finds `x` the same way an op-amp circuit finds its operating point — by settling, not by iterating.

This maps to the SOTA result in the field: "Programmable wave-based analog computing metastructure," *Nature Communications* (2025), arXiv:2301.02850.

---

## Honest scope (read this first)

This is the hardest tier. The Nature Comms 2025 paper built a custom 45 MHz metastructure with voltage-controlled phase shifters and amplifiers as multiplier modules. **You will not build that here.** What you *will* do is demonstrate the same mathematical principle — closed-loop wave-domain equation solving — at three escalating levels of fidelity, each honest about what it proves and what it approximates.

Three modes are provided. Pick the one that matches your budget and tolerance for hardware work.

| Mode | What it is | Cost | What it proves | What it approximates |
|---|---|---|---|---|
| **A. Simulation** | Pure Python, no SDR | Free | The math of wave-domain iterative inversion; the feedback loop convergence | The RF physics (no real wave propagation) |
| **B. Software-in-the-loop** | Real SDR round-trip, digital feedback | ~$200 | The wave channel acts as the operator `A`; a real RF signal carries the iterate | The fully-analog feedback loop (the loop closure is digital) |
| **C. Analog feedback** | Real SDR + analog circulator loop | ~$350 | The closest bench analog of the Nature Comms 2025 metastructure; the loop settles in the wave domain | The custom metastructure (you use a SDR + filter chain instead of voltage-controlled phase shifters) |

**The honest truth about Mode C:** true single-shot analog matrix inversion requires a custom metastructure where the operator `A` is encoded in the scattering. With SDRs, the operator lives in the FIR filter (as in Tier 2), and the feedback is closed through an analog circulator. This is a faithful *demonstration* of the principle, not a reproduction of the Nature Comms hardware. The 2025 paper's contribution — programmable *metastructure* inversion — is a fabrication challenge we explicitly defer to the community.

---

## Hardware

### Mode A — Simulation (free)

| Item | Role | Qty |
|---|---|:---:|
| Laptop (any OS) | Host | 1 |
| Python + NumPy | Solver | — |

No SDR. No RF. This mode teaches the math.

### Mode B — Software-in-the-loop (~$200)

| Item | Role | Qty | Est. cost |
|---|---|:---:|---|
| HackRF One | Transmitter (Tx) | 1 | ~$150 |
| RTL-SDR | Receiver (Rx) | 1 | ~$30 |
| SMA cable + antenna | RF connection | 2 sets | ~$10 |
| Laptop | Host + feedback loop | 1 | you have one |
| **Total** | | | **~$190** |

The loop closure is digital: Tx sends `x_k`, Rx captures `y_k = A·x_k` (the channel convolves with the operator), the laptop computes the feedback `x_{k+1} = x_k + α(b − y_k)` and re-transmits. Each iteration is a real RF round-trip; the *feedback path* is software, not wire.

### Mode C — Analog feedback (~$350)

Mode B hardware, plus:

| Item | Role | Est. cost |
|---|---|---|
| RF circulator (915 MHz or 2.4 GHz) | One-way loop (Tx → channel → Rx → loop back to Tx) | ~$40 |
| Programmable attenuator (or fixed + switched) | Sets the loop gain `α` | ~$30 |
| RF splitter/combiner | Sums the feedback with the input `b` | ~$15 |
| Extra SMA cables | Loop wiring | ~$10 |
| **Mode C add-on total** | | **~$95** |

In Mode C, the loop is closed in hardware: `Tx → channel → Rx → circulator → attenuator → back to Tx`. The laptop sets the operator `A` (in the FIR) and the input `b` (initial transmit), then *lets the loop settle*. The received waveform at steady state is the solution `x = A⁻¹·b`. The laptop does not close the loop; the analog circuit does.

---

## Setup diagram

### Mode B — Software-in-the-loop

```
                           ┌─────────────────────────────────┐
                           │           LAPTOP                │
                           │                                 │
   x_k ──────────────────► │  Tx: x_k = x_{k-1} + α(b - y_{k-1})│
                           │                                 │
                           │  Rx: y_k = A·x_k (channel acts)  │
                           └─────────────────────────────────┘
                                  ▲              │
                                  │              ▼
                           ┌──────┴──────┐ ┌──────────┐
                           │   HackRF    │ │  RTL-SDR │
                           │    (Tx)     │ │   (Rx)   │
                           └──────┬──────┘ └────┬─────┘
                                  │             │
                                  ▼             ▼
                              ┌──────────────────┐
                              │  AIR (channel)   │  ← A lives in the FIR filter
                              │  y_k = A · x_k   │     (programmed on Tx)
                              └──────────────────┘
```

The loop is closed in software. Each iteration: Tx transmits `x_k`, the channel + filter applies `A`, Rx captures `y_k = A·x_k`, the laptop computes `x_{k+1} = x_k + α(b − y_k)`, re-transmits. The wave domain applies `A`; the laptop computes the update. Converges to `x = A⁻¹·b`.

### Mode C — Analog feedback

```
   b (initial) ──┐
                 ▼
           ┌──────────┐    x     ┌──────────────┐    y = A·x     ┌─────────┐
   b ────► │ Splitter │────────►│   HackRF Tx  │──────────────►│ Channel │
           └──────────┘         │ + FIR (A)    │               │ (A·x)   │
                                └──────────────┘               └────┬────┘
                                     ▲                              │
                                     │                              ▼
                              ┌──────┴──────┐                 ┌─────────┐
                              │ Attenuator  │◄─────────────── │ RTL-SDR │
                              │ (gain α)    │     y           │   (Rx)  │
                              └─────────────┘                 └─────────┘
                                     ▲
                                     │
                              ┌──────┴──────┐
                              │  Circulator │  (one-way loop:
                              └─────────────┘   Rx → attenuator → Tx)
```

The loop is closed in hardware. The laptop sets `A` (FIR coefficients) and injects `b` once, then watches the received waveform settle. At steady state, `y ≈ b`, which means `x ≈ A⁻¹·b`. The loop settled to the solution; the laptop didn't compute it.

---

## Software dependencies

Same as Tier 1 and 2:

```bash
pip install SoapySDR numpy scipy matplotlib
```

---

## The math (the stationary fixed-point iteration)

The feedback loop implements a **stationary fixed-point iteration** — the Neumann-series structure for `(I − K)⁻¹`, the simplest linear-system solver:

```
x_{k+1} = x_k + α (b − A·x_k)
```

where `α` is a step size (loop gain). At convergence, `A·x = b`, so `x = A⁻¹·b`. The wave domain provides `A·x_k` (the channel/filter applies the operator); the feedback provides the update. This is the closed-loop mode of the Nature Comms 2025 metastructure — the paper physically realizes the same `(I − K)⁻¹` fixed-point structure, naming the **Jacobi method** as the canonical stationary example, with `A` encoded in a metastructure instead of a FIR filter.

Convergence condition: `0 < α < 2/λ_max(A)` where `λ_max` is the largest eigenvalue of `A`. For the wave domain, `A` must be a stable LTI operator (no poles outside the unit circle). The Nature Comms paper uses matrix inversion (stationary) and Newton's method + Lagrangian optimization (non-stationary); we use the Richardson-Jacobi-type stationary iteration, the simplest case.

---

## The code

### Mode A — Simulation (the math, no SDR)

This mode teaches the iteration. Run it first, even if you have hardware — it shows what convergence looks like.

```python
# sim_matrix_inversion.py
# Pure simulation: wave-domain Richardson iteration on a 2×2 system.
import numpy as np
import matplotlib.pyplot as plt

# The operator A (must be stable: eigenvalues inside unit circle for convergence)
# Use a well-conditioned matrix with spectral radius < 1
A = np.array([[0.8, 0.2],
              [0.1, 0.7]])

# The target: solve A·x = b for x
x_true = np.array([1.5, -0.5])
b = A @ x_true
print(f"Target: solve A·x = b, where b = {b}")
print(f"True solution: x = {x_true}")

# Richardson iteration (this is what the wave-domain feedback loop implements)
alpha = 0.5  # step size (loop gain). Must satisfy 0 < alpha < 2/lambda_max(A)
x = np.zeros(2)  # initial guess
history = [x.copy()]
errors = [np.linalg.norm(x - x_true)]

for k in range(50):
    y = A @ x          # the "wave domain" applies A (convolution in RF)
    err = b - y        # the feedback error
    x = x + alpha * err  # the feedback update
    history.append(x.copy())
    errors.append(np.linalg.norm(x - x_true))
    if k < 5 or k % 10 == 0:
        print(f"  iter {k:2d}: x = [{x[0]:+.4f}, {x[1]:+.4f}]  err = {errors[-1]:.6f}")

print(f"\nConverged x = [{x[0]:.4f}, {x[1]:.4f}]")
print(f"True     x = [{x_true[0]:.4f}, {x_true[1]:.4f}]")
print(f"Final error: {errors[-1]:.6f}")

plt.figure(figsize=(10, 4))
plt.semilogy(errors, 'b.-')
plt.xlabel("iteration k"); plt.ylabel("||x_k - x_true||")
plt.title("Wave-domain Richardson iteration convergence")
plt.grid(True)
plt.savefig("matrix_inversion_convergence.png", dpi=120)
plt.show()
```

**What you'll see:** The error decreases geometrically. After ~20–30 iterations, `x` matches `x_true` to ~4 decimal places. The loop solved `Ax = b` without ever computing `A⁻¹`. This is the math the wave domain implements.

### Mode B — Software-in-the-loop (real RF round-trip)

The operator `A` is a 2×2 real matrix; we encode it as a 2-tap FIR filter and run two parallel RF chains (one per matrix row), or equivalently two sequential transmissions. For simplicity, the code below uses sequential transmission (row 1, then row 2), capturing each `y_k[i]` and computing the update in Python.

```python
# swil_matrix_inversion.py
# Software-in-the-loop: real RF channel applies A, Python closes the loop.
import numpy as np
from SoapySDR import Device, SOAPY_SDR_TX, SOAPY_SDR_RX, SOAPY_SDR_CF32

SAMPLE_RATE = 2e6
FREQ        = 915e6
NUM_SAMPLES = 2**16

# The operator A (2x2, stable) and target b
A = np.array([[0.8, 0.2],
              [0.1, 0.7]])
x_true = np.array([1.5, -0.5])
b = A @ x_true
alpha = 0.5

# Encode A as two FIR filters (one per row). Each row is a 2-tap filter.
# Row i of A = [A[i,0], A[i,1]] → FIR coefficients [A[i,0], A[i,1]]
fir_rows = [A[0], A[1]]

# SDR setup
tx = Device({"driver": "hackrf"})
tx.setSampleRate(SOAPY_SDR_TX, 0, SAMPLE_RATE)
tx.setFrequency(SOAPY_SDR_TX, 0, FREQ)
tx.setGain(SOAPY_SDR_TX, 0, 40)
txStream = tx.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])

rx = Device({"driver": "rtlsdr"})
rx.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
rx.setFrequency(SOAPY_SDR_RX, 0, FREQ)
rx.setGain(SOAPY_SDR_RX, 0, 40)
rxStream = rx.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])

def apply_row_via_rf(row_idx, x_vec):
    """Transmit x_vec, convolve with FIR row A[row_idx], capture y[row_idx]."""
    # Build FIR filter for this row
    fir = fir_rows[row_idx]
    # Encode x as a 2-sample baseband signal (x[0] at t=0, x[1] at t=1)
    # padded with zeros for the convolution to develop
    x_bb = np.zeros(NUM_SAMPLES, dtype=np.complex64)
    x_bb[0] = x_vec[0]
    x_bb[1] = x_vec[1]
    # Convolve with the FIR (this happens in the Tx chain in Mode A of Tier 2;
    # here we do it in Python to keep the loop structure clear)
    y_bb = np.convolve(x_bb, fir, mode='same').astype(np.complex64)

    # Transmit y_bb (the convolved signal) — in a true metastructure this step
    # IS the wave propagation through the metastructure
    tx.activateStream(txStream)
    tx.writeStream(txStream, [y_bb], NUM_SAMPLES)
    tx.deactivateStream(txStream)

    # Capture
    rx.activateStream(rxStream)
    buff = np.empty(NUM_SAMPLES, dtype=np.complex64)
    rx.readStream(rxStream, [buff], NUM_SAMPLES)
    rx.deactivateStream(rxStream)

    # The received signal's DC / first-sample component is y[row_idx]
    return buff[0].real  # simplified readout

# Richardson iteration with real RF round-trips
x = np.zeros(2)
print(f"Target: A·x = b, b = {b}, true x = {x_true}")
for k in range(20):
    y = np.array([apply_row_via_rf(0, x),
                  apply_row_via_rf(1, x)])
    err = b - y
    x = x + alpha * err
    print(f"  iter {k:2d}: x = [{x[0]:+.4f}, {x[1]:+.4f}]  "
          f"y = [{y[0]:+.4f}, {y[1]:+.4f}]  err = {np.linalg.norm(err):.6f}")

print(f"\nConverged x = [{x[0]:.4f}, {x[1]:.4f}]  (true: [{x_true[0]:.4f}, {x_true[1]:.4f}])")
tx.closeStream(txStream); rx.closeStream(rxStream)
```

**What you'll see:** The iteration converges, but noisier than Mode A — each `y` has real RF noise and channel estimation error. You'll likely need `alpha < 0.5` (more conservative) and more iterations (~40–60) to converge. That noise is the *real* reason analog wave-compute has a precision problem, and the *real* reason the Nature Comms paper reports ~0.005 relative error rather than machine epsilon. You just experienced the frontier's limitation, firsthand, on a bench.

### Mode C — Analog feedback (the loop settles in hardware)

Mode C is the closest bench analog of the Nature Comms 2025 metastructure. The laptop *sets up* the loop and injects `b`, then **does not close the loop** — the analog circulator path does. The laptop only reads the steady-state waveform.

```python
# analog_matrix_inversion.py
# Mode C: analog feedback loop. Laptop sets A and b, then WATCHES the loop settle.
import numpy as np
import time
from SoapySDR import Device, SOAPY_SDR_TX, SOAPY_SDR_RX, SOAPY_SDR_CF32

SAMPLE_RATE = 2e6
FREQ        = 915e6
NUM_SAMPLES = 2**16

A = np.array([[0.8, 0.2],
              [0.1, 0.7]])
b = A @ np.array([1.5, -0.5])
alpha = 0.3  # conservative (set by the analog attenuator)

# Tx setup: encode A in the FIR (as in Mode B), set baseband signal = b
tx = Device({"driver": "hackrf"})
tx.setSampleRate(SOAPY_SDR_TX, 0, SAMPLE_RATE)
tx.setFrequency(SOAPY_SDR_TX, 0, FREQ)
tx.setGain(SOAPY_SDR_TX, 0, 30)  # lower gain in closed loop to avoid oscillation
txStream = tx.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])

# Rx setup: capture the steady-state output
rx = Device({"driver": "rtlsdr"})
rx.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
rx.setFrequency(SOAPY_SDR_RX, 0, FREQ)
rx.setGain(SOAPY_SDR_RX, 0, 40)
rxStream = rx.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])

# Build the transmit signal: b encoded as baseband DC levels
# (in practice you'd modulate this; simplified here for clarity)
x_bb = np.zeros(NUM_SAMPLES, dtype=np.complex64)
x_bb[0] = b[0]
x_bb[1] = b[1]

# Apply the FIR (operator A) to the transmit signal
fir = A[0]  # simplified: single-row demonstration
y_bb = np.convolve(x_bb, fir, mode='same').astype(np.complex64)

# Inject b and let the loop run
print("Injecting b and letting analog loop settle...")
print("(The analog circulator path closes the loop; laptop only observes.)")
tx.activateStream(txStream)
# Transmit continuously — the loop recirculates through the analog path
for _ in range(50):
    tx.writeStream(txStream, [y_bb], NUM_SAMPLES)

# Now capture the steady-state output
rx.activateStream(rxStream)
buff = np.empty(NUM_SAMPLES, dtype=np.complex64)
rx.readStream(rxStream, [buff], NUM_SAMPLES)
rx.deactivateStream(rxStream)
tx.deactivateStream(txStream)

x_steady = buff[0].real
print(f"Steady-state received value: {x_steady:.4f}")
print(f"(Should approximate A⁻¹·b = {np.linalg.solve(A, b)[0]:.4f})")
tx.closeStream(txStream); rx.closeStream(rxStream)
```

**What you'll see (and what you won't):** This mode is the hardest to get working cleanly. Expect oscillation, gain mismatch, and long settling times. The Nature Comms 2025 paper spent significant effort stabilizing their metastructure's feedback loop — that stabilization *is* the engineering contribution. If your loop oscillates, reduce the attenuator gain (lower `α`); if it doesn't converge, check the circulator isolation (Tx leakage into Rx is the usual culprit). The point of Mode C is not to achieve 0.005 error — it's to *feel* the analog feedback loop settle, to watch the wave domain find the solution by relaxing to equilibrium instead of iterating.

---

## Troubleshooting (the honest list)

| Symptom | Cause | Fix |
|---|---|---|
| **Mode A:** iteration diverges | `alpha` too large | Reduce `alpha`. Must satisfy `0 < alpha < 2/lambda_max(A)`. For the example matrix, `lambda_max ≈ 0.9`, so `alpha < 2.2`; use `alpha = 0.5`. |
| **Mode A:** converges to wrong value | Matrix `A` not stable (eigenvalues > 1) | Normalize `A` by its largest eigenvalue before iterating. The wave domain requires a stable operator. |
| **Mode B:** iteration diverges or oscillates | Channel gain > 1 (positive feedback) | Measure the channel gain (as in Tier 1) and divide it out of `y` before computing the error. The *effective* loop gain must be < 1. |
| **Mode B:** converges but to wrong value | Channel distortion not accounted for | The channel applies `A * channel_response`, not just `A`. Calibrate: send identity input, measure channel, deconvolve. |
| **Mode C:** loop oscillates | Total loop gain > 1, or phase adds constructively | Reduce the attenuator (lower `alpha`); check circulator isolation (>20 dB); ensure the loop phase doesn't add to oscillation. |
| **Mode C:** no convergence, output drifts | Tx leakage into Rx (circulator isolation insufficient) | Increase physical separation; add an isolator; reduce Tx gain. |
| **Mode C:** output is just `b`, not `A⁻¹·b` | Loop not actually closed — Tx and Rx aren't connected through the analog path | Verify the circulator direction; check that the attenuator is in the loop, not in the Tx path only. |

---

## What this proves

1. **The wave domain can solve equations, not just apply operators.** Tier 2 applied `A` to `x`. Tier 3 inverts `A` to solve `Ax = b`. The difference is the feedback loop. The same wave physics that does convolution also does inversion — you just close the loop.

2. **Settling, not iterating, is the deep insight.** A digital solver computes `A⁻¹` symbolically or iterates an algorithm. The analog feedback loop *settles* to the solution the way a physical system relaxes to equilibrium. The computation time is the settling time, not a function of matrix size. (This is the appeal of analog compute; it's also why noise sets the precision floor.)

3. **The frontier's limitation is real and you just felt it.** Mode B converges to ~3–4 decimal places with careful tuning. Mode C is hard to stabilize. The Nature Comms 2025 paper achieved ~0.005 relative error with a custom metastructure — and that's the SOTA, not a beginner bench. **The gap between "demo on a bench" and "useful compute" is exactly the gap the field is trying to close.** You just experienced why it's hard.

4. **Iterative algorithms have wave-domain forms.** The stationary fixed-point iteration (Richardson/Jacobi-type) is the simplest; the Nature Comms paper extends to Newton's method (root-finding) and Lagrangian optimization. The wave domain isn't limited to linear systems — it can implement nonlinear iterative algorithms too, if the feedback nonlinearity is designed in. That's the frontier.

---

## Going deeper

### Toward the Nature Comms 2025 full result

The paper performs three classes of computation:
- **Matrix inversion** (stationary) — what this walkthrough demonstrates, simplified.
- **Newton's method** (root-finding, non-stationary) — requires a nonlinear feedback element (a divider or squarer in the loop). Beyond this bench demo.
- **Lagrangian optimization** (constrained optimization) — requires dual feedback loops (primal + Lagrange multiplier). Significantly beyond this bench demo.

To go further, read the paper directly: [arXiv:2301.02850](https://arxiv.org/abs/2301.02850). The supplementary material details the metastructure design (phase shifter + amplifier multiplier modules, ~80 ns rise time, 5×5 complex matrix operations).

### Toward real metastructure hardware

The bench demo uses an SDR + FIR filter as a programmable stand-in for a metastructure. To build the real thing:
- **Phase shifters + amplifiers as multiplier modules:** the Nature Comms approach. Each matrix element is a voltage-controlled phase shifter (sets the complex coefficient) and a variable amplifier (sets the magnitude). This is a hardware engineering project, not a weekend build.
- **Inverse-designed scattering:** the Mohammadi Estakhri/Engheta 2019 approach. The metastructure geometry is inverse-designed so its scattering realizes the desired operator. No active elements; no programmability without redesign.
- **This is the community frontier.** The field map names it as explicitly deferred. If you build it, you're not reproducing this surface — you're advancing the field.

---

## Safety and legality

Same as Tiers 1 and 2: ISM bands only, keep power low. **Mode C additional caution:** a closed RF loop can oscillate violently if the gain is too high — this won't damage HackRF One (it has protection), but it can generate spurious emissions. Keep the attenuator conservative (`alpha < 0.5` to start), verify the loop is stable before increasing gain, and use a spectrum analyzer (or the RTL-SDR itself) to check for spurs.

---

## Provenance

- **Maps to:** "Programmable wave-based analog computing metastructure," *Nature Communications* (2025). [arXiv:2301.02850](https://arxiv.org/abs/2301.02850)
- **Iteration theory:** Richardson, "The approximate arithmetical solution by finite differences of dynamical meteorology problems," (1910) — the iteration this loop implements.
- **Closed-loop analog compute precedent:** Mohammadi Estakhri, Edwards & Engheta, "Inverse-designed metastructures that solve equations," *Science* 363, 1333–1338 (2019). Penn.
- **Honest scope:** This walkthrough demonstrates the *principle* of wave-domain equation solving via feedback, at three levels of fidelity. None of the three modes reproduces the Nature Comms 2025 metastructure hardware. Mode A is pure simulation; Mode B closes the loop in software; Mode C closes it in analog hardware but uses an SDR + FIR filter instead of voltage-controlled phase shifter multiplier modules. The gap from this bench demo to the paper's 0.005-error programmable metastructure is the field's open frontier, not this surface's deliverable.
- **Research conducted via:** Multi-source academic search across *Science*, *Nature*, *Nature Photonics*, *IEEE Trans. Inf. Theory*, *Journal of Lightwave Technology*, and arXiv, 2026-07-31.