# Hello World: AirComp Sum

## Tier 1 — The $330 wave-compute primitive

> **The experiment:** Two transmitters send pre-coded signals through the air. The receiver reads their sum from the superposed waveform — without decoding either signal individually. Interference is the operation. The channel is the adder.

This is the simplest wave-compute primitive that exists. If you run this and see the sum appear in your received spectrum, you understand RF-as-compute at a level no paper alone can teach.

It maps to the founding result of over-the-air computation: Nazer & Gastpar, "Compute-and-forward: Harnessing interference through structured codes," *IEEE Trans. Inf. Theory* 57(10), 6463–6486 (2011).

---

## Honest scope

This walkthrough demonstrates **analog superposition AirComp** — the continuous-amplitude version where two RF signals add in the channel and the receiver reads the sum. The full Nazer/Gastpar result uses lattice-coded structured codes to compute functions over finite fields, which is beyond a $330 demo. The analog version teaches the same core insight: **the channel performs the computation; the receiver does not decode the individual inputs.**

If you want the lattice-coded version, see the "Going deeper" section at the end.

---

## Hardware

### Authentic setup (3 RF chains — recommended)

| Item | Role | Qty | Est. cost |
|---|---|:---:|---|
| HackRF One | Transmitter 1 (Tx1) | 1 | ~$150 |
| HackRF One | Transmitter 2 (Tx2) | 1 | ~$150 |
| RTL-SDR (or second HackRF) | Receiver (Rx) | 1 | ~$30 |
| 10 MHz clock cable (BNC) | Frequency sync Tx1↔Tx2 | 1 | ~$5 |
| SMA cables + antennas (2.4 GHz whip) | RF connection | 3 sets | ~$15 |
| Laptop (any OS) | Host | 1 | you have one |
| **Total** | | | **~$350** |

**Why 3 RF chains:** HackRF One is half-duplex — it can either transmit *or* receive at a given time, not both. For true simultaneous AirComp (2 Tx + 1 Rx), you need three units. The RTL-SDR is receive-only and cheap, making it ideal for the Rx role.

### Budget setup (2 RF chains — simplified, less authentic)

| Item | Role | Qty | Est. cost |
|---|---|:---:|---|
| HackRF One | Transmitter (switchable Tx1/Tx2) | 1 | ~$150 |
| RTL-SDR | Receiver | 1 | ~$30 |
| SMA cable + antenna | RF connection | 2 sets | ~$10 |

**Trade-off:** You transmit x1, capture it. Then transmit x2, capture it. Then transmit x1+x2 (pre-computed), capture it. The "channel computation" is done in post-processing by comparing the three captures. You lose the simultaneity that makes AirComp profound, but you learn the signal-processing structure for ~$190. Good for a first pass; upgrade to the authentic setup when you're ready.

---

## Setup diagram (authentic)

```
┌─────────┐     10 MHz sync cable     ┌─────────┐
│ HackRF1 │◄─────────────────────────►│ HackRF2 │
│  (Tx1)  │                           │  (Tx2)  │
└────┬────┘                           └────┬────┘
     │ x1(t) @ f_c                        │ x2(t) @ f_c
     ▼                                    ▼
     │         ┌──────────┐               │
     └────────►│   AIR    │◄──────────────┘
               │ CHANNEL  │
               └────┬─────┘
                    │ h1·x1(t) + h2·x2(t)
                    ▼
               ┌──────────┐
               │  RTL-SDR │
               │   (Rx)   │
               └────┬─────┘
                    │ received sum
                    ▼
               ┌──────────┐
               │  Laptop  │  → visualize f(x1+x2)
               └──────────┘
```

Both transmitters share a 10 MHz clock reference so their carrier frequencies are identical and phase-stable. Without this, the carriers drift relative to each other and the superposition becomes incoherent.

---

## Software dependencies

```bash
# Python 3.9+
pip install SoapySDR numpy scipy matplotlib

# Drivers (OS-specific):
#   HackRF:  hackrf_one USB drivers + libhackrf
#   RTL-SDR: librtlsdr (rtl-sdr package)
#   SoapySDR Vendor plugin: SoapyHackRF, SoapyRTLSDR
```

Verify your SDRs are visible:

```python
import SoapySDR
print(SoapySDR.Device.enumerate())  # should list 2 HackRFs + 1 RTL-SDR
```

---

## The math (one paragraph)

Both transmitters send amplitude-modulated signals at the same carrier frequency `f_c`. If Tx1 sends `x1(t)·cos(2πf_c·t)` and Tx2 sends `x2(t)·cos(2πf_c·t + φ)`, the receiver — after downconversion — sees:

```
r(t) = h1·x1(t) + h2·x2(t)·e^(jφ)
```

where `h1, h2` are channel gains and `φ` is the relative phase offset (zeroed by the shared clock). To get a clean sum `x1(t) + x2(t)`, each transmitter pre-inverts its own channel: Tx1 sends `x1(t)/h1`, Tx2 sends `x2(t)/h2`. The receiver then sees:

```
r(t) = x1(t) + x2(t)
```

**This is channel-inversion pre-coding — the standard AirComp technique.** The transmitters don't need to know each other's signals; they only need to know their own channel to the receiver. The channel does the sum. The receiver never decodes x1 or x2 individually.

---

## The code

### Step 0 — Channel estimation (run once, per transmitter)

Each transmitter sends a known pilot tone; the receiver measures `h1` and `h2`.

```python
# channel_estimate.py
# Run once for each transmitter to measure its channel gain.
import numpy as np
from SoapySDR import Device, SOAPY_SDR_RX, SOAPY_SDR_CF32

SAMPLE_RATE = 2e6   # 2 Msps
FREQ        = 915e6  # 915 MHz ISM band (use 2.4 GHz if your antennas support it)
NUM_SAMPLES = 2**18

# Pilot: a simple known BPSK sequence
pilot = np.array([1, -1, 1, -1] * (NUM_SAMPLES // 4), dtype=np.complex64)

rx = Device({"driver": "rtlsdr"})
rx.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
rx.setFrequency(SOAPY_SDR_RX, 0, FREQ)

rxStream = rx.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
rx.activateStream(rxStream)

# Tell ONE transmitter to send the pilot, then capture:
buff = np.empty(NUM_SAMPLES, dtype=np.complex64)
sr = rx.readStream(rxStream, [buff], NUM_SAMPLES)
rx.deactivateStream(rxStream)
rx.closeStream(rxStream)

# Channel gain = correlation of received signal with known pilot
h = np.mean(buff[:NUM_SAMPLES] * np.conj(pilot))
print(f"Channel gain h = {h}  |h| = {abs(h):.4f}  phase = {np.angle(h):.4f} rad")
# Save h for use in the transmitters' pre-coding
np.save("h1.npy", h)  # or h2.npy when measuring Tx2
```

### Step 1 — Transmitter (run two copies, one per HackRF)

```python
# tx_aircomp.py  — run as: python tx_aircomp.py --node 1 --value 3.0
# Tx1 sends  x1(t)/h1   where x1(t) = VALUE (a constant amplitude)
import numpy as np
import argparse
from SoapySDR import Device, SOAPY_SDR_TX, SOAPY_SDR_CF32

SAMPLE_RATE = 2e6
FREQ        = 915e6
NUM_SAMPLES = 2**18
DURATION_S  = 2.0   # transmit for 2 seconds

p = argparse.ArgumentParser()
p.add_argument("--node", type=int, required=True)   # 1 or 2
p.add_argument("--value", type=float, required=True) # the operand: x1 or x2
args = p.parse_args()

# Load this transmitter's channel gain (measured in Step 0)
h = np.load(f"h{args.node}.npy")

# Pre-coding: divide by channel gain so the receiver sees the clean sum
signal_value = args.value / h  # channel inversion

# Build the baseband waveform: a constant complex amplitude
# (in a real demo you'd modulate this with a sequence; here we send a DC tone)
samples = np.full(NUM_SAMPLES, signal_value, dtype=np.complex64)

tx = Device({"driver": "hackrf"})
tx.setSampleRate(SOAPY_SDR_TX, 0, SAMPLE_RATE)
tx.setFrequency(SOAPY_SDR_TX, 0, FREQ)
tx.setGain(SOAPY_SDR_TX, 0, 40)  # adjust to avoid clipping

txStream = tx.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0])
tx.activateStream(txStream)

# Transmit continuously for DURATION_S
num_blocks = int(DURATION_S * SAMPLE_RATE / NUM_SAMPLES)
for _ in range(num_blocks):
    sr = tx.writeStream(txStream, [samples], NUM_SAMPLES)

tx.deactivateStream(txStream)
tx.closeStream(txStream)
print(f"Tx{args.node} sent value {args.value} (pre-coded as {signal_value:.4f})")
```

### Step 2 — Receiver (run on the laptop with RTL-SDR)

```python
# rx_aircomp.py  — captures and displays the sum
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

print("Listening for AirComp sum... (start both transmitters now)")

buff = np.empty(NUM_SAMPLES, dtype=np.complex64)
sr = rx.readStream(rxStream, [buff], NUM_SAMPLES)
rx.deactivateStream(rxStream)
rx.closeStream(rxStream)

# The received signal should be approximately x1 + x2
# Downconvert and average to read the DC component (the constant-amplitude sum)
received_sum = np.mean(buff)
print(f"Received: {received_sum.real:.4f} + {received_sum.imag:.4f}j")
print(f"|received| ≈ {abs(received_sum):.4f}")
print(f"(This should be approximately x1 + x2)")

# Plot the received spectrum
plt.psd(buff, NFFT=1024, Fs=SAMPLE_RATE/1e6, Fc=FREQ/1e6)
plt.xlabel("Frequency (MHz)")
plt.title("Received AirComp spectrum — the peak is x1+x2")
plt.savefig("aircomp_received.png", dpi=120)
plt.show()
```

---

## What you should see

1. **Run Step 0** for each transmitter (one at a time). Save `h1.npy` and `h2.npy`. These are your channel gains.

2. **Start the receiver** (`rx_aircomp.py`). It waits and listens.

3. **Start both transmitters simultaneously** in two terminals:
   ```bash
   python tx_aircomp.py --node 1 --value 3.0   # Tx1 sends x1 = 3.0
   python tx_aircomp.py --node 2 --value 5.0   # Tx2 sends x2 = 5.0
   ```

4. **The receiver prints:**
   ```
   |received| ≈ 8.0
   (This should be approximately x1 + x2)
   ```
   **3 + 5 = 8.** The channel computed the sum. You never decoded x1 or x2 individually. The interference *was* the addition.

5. **The spectrum plot** shows a single peak at the carrier frequency — that peak's amplitude is the sum.

---

## Troubleshooting (the honest list)

| Symptom | Cause | Fix |
|---|---|---|
| Sum is wrong / noisy | Carriers drifting apart | Connect the 10 MHz clock sync cable between both HackRFs. Without it, the carriers drift and superposition is incoherent. |
| Sum is ~0 instead of x1+x2 | Phase cancellation | Channel estimation gave wrong phase. Re-run Step 0; ensure the transmitter wasn't moved between estimation and the experiment. |
| Received signal is clipped | Tx gain too high | Reduce `setGain(SOAPY_SDR_TX, 0, ...)` to 20–30 dB. Watch the received amplitude in Step 0. |
| RTL-SDR sees nothing | Wrong frequency or gain | Check `FREQ` matches; set RTL-SDR gain to 40–50 dB. RTL-SDR has a DC spike at the center frequency — offset-tune by ±100 kHz if it obscures the signal. |
| DC offset dominates | RTL-SDR direct conversion artifact | Offset-tune the receiver by 100–200 kHz and shift the transmit frequency to match, or use a non-DC modulation (BPSK sequence instead of constant amplitude). |
| "No devices found" | Driver not installed | Install libhackrf + SoapyHackRF plugin (HackRF) and librtlsdr + SoapyRTLSDR plugin (RTL-SDR). |

---

## What this proves

Read this carefully — it's the point of the whole experiment.

1. **The channel is the adder.** The receiver read `x1 + x2` without ever decoding `x1` or `x2` individually. The addition happened in the air, in the wave domain, at the speed of light.

2. **Interference is computation, not noise.** Conventional wireless treats two transmitters on the same frequency as a collision to be avoided. AirComp treats it as a function evaluation. Same physics; opposite interpretation.

3. **The transmitters cooperated without sharing data.** Each transmitter only knew its own channel gain and its own value. The channel did the combining. This is the foundation of "computing while communicating" — the 6G research frontier.

4. **This is analog.** The precision is limited by noise, channel estimation error, and hardware imperfections. You'll get ~8.0 ± some error. That error is *the* reason AirComp with digital lattice codes (the full Nazer/Gastpar result) exists — and it's the reason pure analog wave-compute has a scaling problem. You just experienced it firsthand.

---

## Going deeper

### Toward the real Nazer/Gastpar result

The experiment above uses analog amplitude superposition. The actual Nazer & Gastpar 2007/2011 result uses **lattice-coded structured codes** to compute functions over finite fields, which is noise-resilient and exact (not approximate). The bridge:

- Instead of sending `x1/h1` as a continuous amplitude, you send a lattice codeword whose structure guarantees the receiver recovers the finite-field sum.
- This requires nested lattice codes — the construction is in the paper's Section IV.
- A software implementation of compute-and-forward lattice coding is a substantial project in itself. It's a natural "Tier 1.5" extension if this surface grows.

### Toward Tier 2 (wave-domain convolution)

The AirComp sum is a single operation (addition). Tier 2 upgrades the operator: instead of the channel computing `x1 + x2`, you program a filter chain that computes `conv(x, h)` — convolution — in the wave domain. That's the Silva et al. 2014 result. See `docs/hello-world-convolution.md` *(coming)*.

### Toward Tier 3 (matrix inversion via feedback)

Tier 3 closes the loop: the receiver feeds its output back to the transmitter, and the feedback loop solves `Ax = b` for `x` in the wave domain. That's the *Nature Communications* 2025 result. See `docs/hello-world-matrix-inversion.md` *(coming)*.

---

## Safety and legality

- **ISM bands only.** Use 915 MHz (US) or 2.4 GHz (global). These are license-free bands designated for low-power experimentation.
- **Keep power low.** HackRF One max output is ~15 dBm (~30 mW). That's well within ISM limits. Don't add external amplifiers.
- **Check your local regulations** if you're outside the US. ISM band rules vary by country.
- **Don't transmit near sensitive equipment** (medical, aviation). A 30 mW signal at 915 MHz is harmless to humans but can interfere with unshielded electronics.

---

## Provenance

- **Maps to:** Nazer & Gastpar, "Compute-and-forward: Harnessing interference through structured codes," *IEEE Trans. Inf. Theory* 57(10), 6463–6486 (2011). [DOI: 10.1109/TIT.2011.2165816](https://doi.org/10.1109/TIT.2011.2165816)
- **Origin result:** Nazer & Gastpar, "Computation over multiple-access channels," *IEEE Trans. Inf. Theory* 53(10), 3498–3516 (2007).
- **Survey:** "Over-the-Air Computation for 6G," arXiv:2210.10524 (2022) `[preprint]`
- **Honest scope:** This walkthrough demonstrates the analog superposition principle. The full lattice-coded result is not reproduced here. The analog version is pedagogically faithful to the core insight ("interference is computation") but does not achieve the noise-resilience or exactness of the finite-field version.
- **Research conducted via:** Multi-source academic search across *Science*, *Nature*, *Nature Photonics*, *IEEE Trans. Inf. Theory*, *Journal of Lightwave Technology*, and arXiv, 2026-07-31.