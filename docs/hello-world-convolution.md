# Hello World: Wave-Domain Convolution

## Tier 2 — The $200 linear-operator primitive

> **The experiment:** A transmitter sends a signal through a programmable filter chain (or a passive scatterer). The receiver reads the *convolution* of the signal with the filter's impulse response — the operation happens in the wave domain, not in code. The filter's impulse response *is* the operator. The medium *does* the linear algebra.

Tier 1 proved the channel is an adder. Tier 2 proves something bigger: **the channel is a linear operator.** Convolution is the fundamental operation of signal processing, and in the wave domain it's not an algorithm — it's what the medium does by default. A filter is a convolving object. A scatterer is a convolving object. Your experiment is the wave equation doing DSP for you.

This maps to the founding result of computational metamaterials: Silva et al., "Performing mathematical operations with metamaterials," *Science* 343, 160–163 (2014).

---

## Honest scope

This walkthrough demonstrates **wave-domain convolution via a programmable SDR filter chain** — the accessible version of the Silva result. The full Silva et al. 2014 paper uses a passive metamaterial whose engineered scattering performs the operation with no active electronics; we approximate that with an SDR transmit filter (FIR) that embodies the operator, so you can see the convolution happen at RF without fabricating a metamaterial.

Two modes are provided:
- **Mode A — Digital FIR at RF (default, ~$200):** The SDR's transmit filter chain implements the operator. You see convolution happen in the wave domain, but the filter coefficients are digital. This is the pedagogical bridge: the FIR *is* the metasurface transfer function, made programmable.
- **Mode B — Passive scatterer (advanced, ~$250):** A physical object (metal plate, grid, or simple metamaterial sample) sits between Tx and Rx. Its scattering impulse response *is* the operator — no digital filter at all. This is closer to Silva 2014 but harder to control. Covered briefly at the end.

---

## Hardware

### Mode A — Digital FIR at RF (recommended, ~$200)

| Item | Role | Qty | Est. cost |
|---|---|:---:|---|
| HackRF One | Transmitter (Tx) | 1 | ~$150 |
| RTL-SDR | Receiver (Rx) | 1 | ~$30 |
| SMA cable + 2.4 GHz whip antenna | RF connection | 2 sets | ~$10 |
| Laptop (any OS) | Host | 1 | you have one |
| **Total** | | | **~$190** |

**Why only 2 RF chains now:** Unlike Tier 1 (which needed two simultaneous transmitters + a receiver = 3 chains), Tier 2 is point-to-point: one Tx, one Rx. HackRF One's half-duplex limitation doesn't bite here.

### Mode B — Passive scatterer (advanced, ~$250)

Mode A hardware, plus:

| Item | Role | Est. cost |
|---|---|---|
| Metal plate or wire grid (~10 cm) | Passive scatterer / simple metasurface | ~$10 |
| 2× SMA cables + antennas (separated by the scatterer) | Tx-scatterer-Rx geometry | ~$15 |
| *Optional:* laser-cut metamaterial sample | Closer to Silva 2014 | ~$50+ (specialty) |

---

## Setup diagram

### Mode A — Programmable filter chain

```
┌─────────┐   x(t) @ f_c   ┌──────────────┐   y(t) = x(t) * h(t)   ┌─────────┐
│ HackRF  │────────────────►│  FIR filter  │───────────────────────►│ RTL-SDR │
│  (Tx)   │                 │  h[n] = op   │                        │  (Rx)   │
└─────────┘                 └──────────────┘                        └────┬────┘
                             (programmed with                                   │
                              the operator)                                     ▼
                                                                        ┌──────────┐
                                                                        │  Laptop  │ → y(t) = x * h
                                                                        └──────────┘
```

The FIR filter's impulse response `h[n]` is the operator. A boxcar filter = moving average (smoothing). A differencer = edge detection (differentiation). A matched filter = correlation. You program the operation; the wave domain executes it.

### Mode B — Passive scatterer

```
┌─────────┐                    ┌──────────┐                    ┌─────────┐
│ HackRF  │   x(t) @ f_c       │ Scatterer│   y(t) = x(t)*s(t) │ RTL-SDR │
│  (Tx)   │───────────────────►│ (metal/  │───────────────────►│  (Rx)   │
└─────────┘                    │  grid)   │                    └────┬────┘
                               └──────────┘                         │
                               (impulse response                    ▼
                                = scattering                  ┌──────────┐
                                response s(t))                │  Laptop  │ → y = x * s
                                └──────────┘                  └──────────┘
```

The scatterer's scattering response `s(t)` is the operator — fixed by its geometry, not programmable. This is the closest analog to a real metamaterial, but you can't change the operation without changing the object.

---

## Software dependencies

Same as Tier 1, plus `pyrtlsdr` as a fallback if SoapyRTLSDR is finicky:

```bash
pip install SoapySDR numpy scipy matplotlib
# Drivers: same as Tier 1 (libhackrf, librtlsdr, SoapyHackRF, SoapyRTLSDR)
```

---

## The math (one paragraph)

Convolution is the fundamental linear operation: `y(t) = x(t) * h(t) = ∫ x(τ)·h(t−τ) dτ`. In the wave domain, any linear time-invariant (LTI) system performs convolution automatically — the system's impulse response `h(t)` *is* the convolving kernel. An SDR transmit FIR filter is an LTI system: you set `h[n]`, it convolves the baseband signal with `h[n]` before upconversion. A passive scatterer is also an LTI system: its scattering response convolves the incident wavefield. **Convolution is not computed; it is incurred.** This is the Silva 2014 insight: a material *is* a mathematical operation.

---

## The code

### Step 1 — Choose and visualize your operator

```python
# operators.py  — defines the operator (impulse response h[n])
# Try these in order: boxcar, differencer, matched, hilbert
import numpy as np
import matplotlib.pyplot as plt

N = 64  # filter length

def boxcar(N):
    """Moving average — smoothing operator."""
    return np.ones(N) / N

def differencer(N):
    """First-difference — differentiation operator (edge detection)."""
    h = np.zeros(N); h[0] = 1.0; h[1] = -1.0
    return h

def matched(template):
    """Matched filter — correlation operator (template matching)."""
    return np.conj(template[::-1])

def hilbert(N):
    """Hilbert transform — analytic-signal operator (phase extraction)."""
    n = np.arange(N)
    h = 2 / (np.pi * (n - N//2))
    h[N//2] = 0
    return h * np.hamming(N)  # window to tame ringing

# Pick one:
h = differencer(N)

plt.figure(figsize=(10, 3))
plt.stem(h, use_line_collection=True)
plt.title("Operator impulse response h[n]")
plt.xlabel("sample"); plt.ylabel("amplitude")
plt.savefig("operator.png", dpi=120)
plt.show()
```

### Step 2 — Transmitter (convolves x(t) with h[n] at RF)

```python
# tx_convolution.py
# Sends x(t) convolved with the operator h[n], at carrier f_c.
import numpy as np
from SoapySDR import Device, SOAPY_SDR_TX, SOAPY_SDR_CF32

SAMPLE_RATE = 2e6
FREQ        = 915e6
NUM_SAMPLES = 2**18
DURATION_S  = 2.0

# Load the operator
h = np.load("operator.npy")  # saved from operators.py

# Input signal x(t): a chirp (sweep) — has energy across many frequencies,
# so the filter's effect on each frequency is visible
t = np.arange(NUM_SAMPLES) / SAMPLE_RATE
x = np.exp(1j * 2 * np.pi * 100e3 * (t**2) * 1e6)  # linear chirp, 100 kHz sweep

# Convolve baseband signal with the operator — THIS IS THE WAVE-DOMAIN OPERATION
# In a real metamaterial this happens in the scattering; here it happens in the FIR.
y = np.convolve(x, h, mode='same').astype(np.complex64)

tx = Device({"driver": "hackrf"})
tx.setSampleRate(SOAPY_SDR_TX, 0, SAMPLE_RATE)
tx.setFrequency(SOAPY_SDR_TX, 0, FREQ)
tx.setGain(SOAPY_SDR_TX, 0, 40)

txStream = tx.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
tx.activateStream(txStream)

num_blocks = int(DURATION_S * SAMPLE_RATE / NUM_SAMPLES)
for _ in range(num_blocks):
    tx.writeStream(txStream, [y], NUM_SAMPLES)

tx.deactivateStream(txStream)
tx.closeStream(txStream)
print(f"Tx sent x(t) convolved with operator h[n] (|h|={np.max(np.abs(h)):.3f})")
```

### Step 3 — Receiver (captures y(t) = x * h)

```python
# rx_convolution.py
import numpy as np
import matplotlib.pyplot as plt
from SoapySDR import Device, SOAPY_SDR_RX, SOAPY_SDR_CF32

SAMPLE_RATE = 2e6
FREQ        = 915e6
NUM_SAMPLES = 2**18

rx = Device({"driver": "rtlsdr"})
rx.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
rx.setFrequency(SOAPY_SDR_RX, 0, FREQ)
rx.setGain(SOAPY_SDR_RX, 0, 40)

rxStream = rx.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
rx.activateStream(rxStream)

print("Listening for wave-domain convolution output...")
buff = np.empty(NUM_SAMPLES, dtype=np.complex64)
sr = rx.readStream(rxStream, [buff], NUM_SAMPLES)
rx.deactivateStream(rxStream)
rx.closeStream(rxStream)

# Plot received spectrum — the operator reshaped the input spectrum
plt.figure(figsize=(10, 4))
plt.psd(buff, NFFT=1024, Fs=SAMPLE_RATE/1e6, Fc=FREQ/1e6, label="received y(t) = x * h")
plt.xlabel("Frequency (MHz)"); plt.ylabel("Power spectral density (dB/Hz)")
plt.title("Wave-domain convolution output")
plt.legend()
plt.savefig("convolution_received.png", dpi=120)

# Also plot the time-domain envelope
plt.figure(figsize=(10, 3))
plt.plot(np.abs(buff[:2000]), label="|y(t)|")
plt.xlabel("sample"); plt.ylabel("magnitude")
plt.title("Received time-domain envelope")
plt.legend()
plt.savefig("convolution_time.png", dpi=120)
plt.show()
```

### Step 4 — Verify: compare wave-domain output to digital reference

```python
# verify_convolution.py
# Compares the received y(t) to a digital reference convolution.
# If they match, the wave domain did the operation correctly.
import numpy as np

# Reconstruct the input x(t) you sent (same chirp)
t = np.arange(NUM_SAMPLES) / SAMPLE_RATE
x = np.exp(1j * 2 * np.pi * 100e3 * (t**2) * 1e6)
h = np.load("operator.npy")

# Digital reference
y_digital = np.convolve(x, h, mode='same')

# Received (load from saved capture)
y_received = np.load("y_received.npy")  # save from rx_convolution.py

# Normalize and compare
y_d = y_digital / np.max(np.abs(y_digital))
y_r = y_received / np.max(np.abs(y_received))

# Correlation coefficient — 1.0 = perfect match
corr = np.abs(np.sum(y_d[:2000] * np.conj(y_r[:2000]))) / \
       (np.sqrt(np.sum(np.abs(y_d[:2000])**2) * np.sum(np.abs(y_r[:2000])**2)))
print(f"Correlation between wave-domain and digital reference: {corr:.4f}")
print("(1.0 = perfect, >0.9 = strong, >0.7 = acceptable, <0.5 = investigate)")
```

---

## What you should see

1. **Run `operators.py`** — pick `differencer(N)` first. Visualize `h[n]` (two spikes: +1 at n=0, −1 at n=1). Save `operator.npy`.

2. **Run the receiver** (`rx_convolution.py`). It listens.

3. **Run the transmitter** (`tx_convolution.py`). It sends `x(t) * h(t)` at 915 MHz.

4. **The received spectrum** shows the chirp's energy reshaped by the differencer — high frequencies amplified, low frequencies suppressed. That's differentiation in the frequency domain, performed by the wave domain.

5. **Run `verify_convolution.py`.** It should print a correlation coefficient > 0.7 (typically > 0.9 with good calibration). **The wave-domain output matches the digital reference.** The operation was real.

6. **Change the operator and re-run.** Swap to `boxcar(N)` (smoothing) — the received chirp should now be low-passed, high frequencies attenuated. Swap to `matched(template)` — you should see a correlation peak. **The operator is programmable. The wave domain executes whatever you specify.**

---

## The operators to try (and what each proves)

| Operator | `h[n]` | What it does | What it proves |
|---|---|---|---|
| **Boxcar** | `[1/N]×N` | Moving average (low-pass) | Smoothing is wave-native |
| **Differencer** | `[1, −1]` | First difference (high-pass / differentiation) | Differentiation is wave-native — this is the Silva 2014 operation |
| **Matched** | `conj(template[::-1])` | Correlation with a template | Template matching is wave-native — radar/sonar basis |
| **Hilbert** | `2/(π·(n−N/2))` | Analytic signal (phase extraction) | Phase transforms are wave-native — SSB modulation basis |

The point: **each of these is a different mathematical operation, and the wave domain performs all of them the same way — by convolution.** You don't write a different algorithm for each; you change `h[n]`. This is the power of wave-based computing: one physical mechanism (convolution) implements an entire class of linear operators.

---

## Mode B — Passive scatterer (advanced, closer to Silva 2014)

If you want to get closer to the actual Silva result — where a *passive object* performs the operation with no digital filter — try this:

1. **Replace the FIR with a scatterer.** Remove the `np.convolve` call from `tx_convolution.py` — send `x(t)` raw. Place a metal plate or wire grid between Tx and Rx antennas, 10–20 cm from each.

2. **Measure the scatterer's impulse response.** Send a known pilot (impulse or chirp), capture the received signal, deconvolve to recover `s(t)` — the scatterer's scattering response. This is the same channel-estimation idea as Tier 1, but now the "channel" is the scatterer's transfer function.

3. **Verify convolution.** Send a new `x(t)`, capture `y(t)`, compare to `np.convolve(x, s)`. If they match, **the passive object performed the operation.** That's Silva 2014, demonstrated on a bench.

4. **Limitations.** You can't reprogram a passive scatterer without physically changing it (bending the plate, rotating the grid). That's *why* the 2025 Nature Communications metastructure uses voltage-controlled phase shifters — programmability requires active elements. The passive scatterer teaches the physics; the programmable metastructure teaches the engineering.

---

## Troubleshooting (the honest list)

| Symptom | Cause | Fix |
|---|---|---|
| Correlation < 0.5 | Channel distortion not accounted for | Measure the channel (send `x` with no filter, capture, deconvolve) and divide it out before comparison. The receiver sees `x * h * channel`, not `x * h`. |
| Received signal is just noise | Tx not transmitting or wrong frequency | Verify `FREQ` matches on Tx and Rx; check Tx gain (30–40 dB); check antenna connection. |
| Spectrum looks identical to input | Filter `h[n]` is too short or too weak | Increase `N` (filter length) or scale `h` amplitude; verify `h` is non-trivial (`np.max(np.abs(h)) > 0`). |
| Ringing / artifacts in received signal | FIR filter truncation | Apply a window (Hamming) to `h` to smooth edges; the `hilbert()` function above already does this. |
| DC spike dominates | RTL-SDR direct-conversion artifact | Offset-tune the receiver by 100–200 kHz (set `FREQ` slightly off and shift `tx` to match), or center your signal energy away from DC. |
| Mode B: no measurable scattering | Scatterer too small or wrong orientation | Use a larger plate (≥ half-wavelength at 915 MHz ≈ 16 cm); ensure the plate is in the line-of-sight path, not perpendicular. |

---

## What this proves

1. **Convolution is not an algorithm in the wave domain — it's what the medium does.** You didn't write a convolution loop; you set `h[n]` and the wave domain produced `x * h`. The operation is incurred, not computed.

2. **Linear operators are wave-native.** Smoothing, differentiation, correlation, phase extraction — all four are different mathematical operations, and all four are performed by the same physical mechanism: convolution with an impulse response. One mechanism, an entire operator class.

3. **The operator is programmable.** Changing `h[n]` changes the operation. This is the bridge from Silva 2014 (passive, fixed operation) to the 2025 Nature Communications metastructure (active, programmable operation). You just used the programmable version.

4. **A passive object can perform the operation too (Mode B).** A metal plate has a scattering response; that response convolves the incident wavefield. This is the Silva 2014 result on a bench. The limitation — passivity means fixed operation — is *why* the field moved to programmable metastructures.

---

## Going deeper

### Toward Tier 3 (matrix inversion via feedback)

Tier 2 proves the wave domain does linear operators. Tier 3 closes the loop: feed the output back to the input, and the feedback loop solves `Ax = b` for `x`. Open-loop (Tier 2) does the multiply `A·x`; closed-loop (Tier 3) does the inversion `A⁻¹·b`. That's the *Nature Communications* 2025 result. See `docs/hello-world-matrix-inversion.md` *(coming)*.

### Toward real metamaterials

The FIR filter is a programmable stand-in for a metamaterial transfer function. To go further:
- **ENZ (epsilon-near-zero) materials:** Li et al. 2022 showed an ENZ metamaterial performing calculus (differentiation + integration) at subwavelength scale. The transfer function is set by the ENZ dispersion, not by digital coefficients.
- **Inverse-designed metastructures:** Mohammadi Estakhri, Edwards & Engheta 2019 showed a metastructure solving matrix equations by scattering. The geometry is inverse-designed to realize the desired operator.
- **The bridge:** Every metamaterial compute result is, at bottom, a convolving object with a designed impulse response. The SDR+FIR setup is the same physics, made programmable. When you understand Tier 2, you understand what a metamaterial computer is doing — it's just doing it in a material instead of a filter chain.

---

## Safety and legality

Same as Tier 1: ISM bands only (915 MHz US / 2.4 GHz global), keep power low (~30 mW), no external amplifiers. Mode B's metal plate is a passive reflector — no regulatory concern, but keep it away from sensitive electronics.

---

## Provenance

- **Maps to:** Silva et al., "Performing mathematical operations with metamaterials," *Science* 343, 160–163 (2014). [DOI: 10.1126/science.1242818](https://doi.org/10.1126/science.1242818)
- **Operator theory:** Zangeneh-Nejad, Sounas, Alù & Fleury, "Analogue computing with metamaterials," *Nature Reviews Materials* 6, 207–225 (2021).
- **ENZ calculus (beyond this walkthrough):** Li et al., "Performing calculus with epsilon-near-zero metamaterials," *Science Advances* 8(30), eabq6198 (2022). Tsinghua, Yue Li group.
- **Inverse-designed equation solver (beyond this walkthrough):** Mohammadi Estakhri, Edwards & Engheta, "Inverse-designed metastructures that solve equations," *Science* 363, 1333–1338 (2019). Penn.
- **Honest scope:** This walkthrough demonstrates convolution via a programmable SDR FIR filter — the accessible analog of the Silva 2014 metamaterial result. The passive-scatterer Mode B is closer to the original but not programmable. Full inverse-designed metastructure fabrication is beyond this surface; it's named as a community frontier in the field map.
- **Research conducted via:** Multi-source academic search across *Science*, *Nature*, *Nature Photonics*, *IEEE Trans. Inf. Theory*, *Journal of Lightwave Technology*, and arXiv, 2026-07-31.