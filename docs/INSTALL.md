# Installation

## 60-second quick start (simulation, no hardware)

```bash
git clone https://github.com/rf-compute/rf-compute.git
cd rf-compute
pip install -e .
python examples/tier1_aircomp_kernel.py    # AirComp: 3+5=8
python examples/tier2_convolution_kernel.py # 4 operators
python examples/tier3_inversion_kernel.py   # solves Ax=b
```

That's it. The simulation backend runs on NumPy alone — no SDR hardware, no RF drivers, no special permissions. You should see:

```
AirComp sum: 3.0 + 5.0 = 8.0
```

If you see that, the kernel works. The rest of this document is for when you want to run the real-RF walkthroughs with SDR hardware.

---

## Three install paths

| Path | What you get | Cost | When to use |
|---|---|---|---|
| **Simulation only** | `pip install -e .` | Free | First run, learning the API, Tier 3 Mode A |
| **+ SDR backend** | `pip install -e ".[sdr]"` | Free software | You have HackRF/RTL-SDR hardware |
| **+ Visualization** | `pip install -e ".[viz]"` | Free software | Running the walkthrough scripts that plot spectra |
| **+ Dev** | `pip install -e ".[dev]"` | Free software | Contributing (includes pytest) |

You can combine extras: `pip install -e ".[sdr,viz]"` installs both SDR and visualization support.

---

## Path 1 — Simulation only (default)

```bash
pip install -e .
```

Dependencies installed: `numpy` only. This is the zero-friction path — runs on any laptop, any OS, no hardware. Use this to:
- Learn the kernel API (`apply`, `solve`)
- Run Tier 3 Mode A (matrix inversion simulation)
- Develop new operators without hardware
- Run CI tests

The simulation backend is **mathematically faithful to the wave-domain operations** (the channel computes the sum, the medium convolves the signal, the feedback loop solves the equation) but it doesn't model RF physics — no noise, no channel distortion, no carrier drift. For that, you need hardware.

---

## Path 2 — SDR backend (real hardware)

```bash
pip install -e ".[sdr]"
```

This installs the `SoapySDR` Python binding. But **the SDR drivers themselves are not Python packages** — they're system libraries that talk to the hardware. You must install them separately, by OS.

### Windows

#### HackRF One drivers

1. Download the HackRF Windows drivers from [the HackRF release page](https://github.com/greatscottgadgets/hackrf/releases).
2. Extract and run `hackrf-windows-installer.exe`.
3. Plug in your HackRF One. Windows should detect it; if prompted for a driver, point it to the extracted driver folder.

#### RTL-SDR drivers

1. Download `rtl-sdr-win32-x64.zip` from [the osmocom RTL-SDR page](https://osmocom.org/projects/rtl-sdr).
2. Extract to a permanent location (e.g., `C:\rtl-sdr\`).
3. Add that directory to your `PATH`:
   ```powershell
   setx PATH "%PATH%;C:\rtl-sdr"
   ```
4. Plug in your RTL-SDR. If Windows asks for a driver, use Device Manager → Update Driver → Browse → point to the extracted `rtl-sdr` folder.

#### SoapySDR + vendor plugins

The easiest path on Windows is [PothosSDR](https://github.com/pothosware/pothos/wiki/PothosSDR-Install-Recipes), which bundles SoapySDR + HackRF + RTL-SDR support in one installer:

1. Download the latest PothosSDR installer from [the Pothos downloads page](https://downloads.myriadrf.org/project/PothosSDR/).
2. Run the installer. It installs SoapySDR, SoapyHackRF, SoapyRTLSDR, and the underlying drivers.
3. Restart your terminal so the new `PATH` entries are picked up.

Verify:

```bash
python -c "from SoapySDR import Device; print(Device.enumerate())"
```

If you see a list of devices (your HackRF and/or RTL-SDR), you're ready. If you see `[]`, the drivers are installed but no SDR is plugged in. If you see `ImportError`, the SoapySDR Python binding isn't on your path — check the PothosSDR install.

### macOS

```bash
brew install hackrf rtl-sdr soapysdr
brew install soapysdr-plugins-all   # or soapysdr-plugins-hackrf / soapysdr-plugins-rtlsdr
pip install -e ".[sdr]"
```

### Linux (Debian/Ubuntu)

```bash
sudo apt install hackrf libhackrf-dev rtl-sdr librtlsdr-dev soapysdr-tools \
                 python3-soapysdr soapysdr-module-hackrf soapysdr-module-rtlsdr
pip install -e ".[sdr]"
```

### Linux (Fedora)

```bash
sudo dnf install hackrf rtl-sdr soapysdr SoapySDR-devel python3-SoapySDR
pip install -e ".[sdr]"
```

### Verify the SDR backend

```bash
python -c "
from rf_compute import WaveComputeKernel
k = WaveComputeKernel(backend='sdr')
print('Backend:', k.backend, '(degraded:' , k._degraded, ')')
"
```

- If `backend: sdr (degraded: False)` — SDR backend is live.
- If `backend: sim (degraded: True)` — SoapySDR is installed but not found, or no SDR is attached. The kernel fell back to simulation (this is the graceful degradation). Check your driver install and plug in the SDR.

---

## Path 3 — Visualization

The walkthrough scripts in `docs/hello-world-*.md` use `matplotlib` for spectrum plots and convergence charts. The kernel itself doesn't depend on matplotlib, so it's an optional extra:

```bash
pip install -e ".[viz]"
```

---

## Path 4 — Development (contributors)

If you're contributing (see `CONTRIBUTING.md`):

```bash
pip install -e ".[dev]"
pytest
```

This installs pytest, SoapySDR, and matplotlib together. Run the test suite with `pytest` from the repo root. (The test suite is a future deliverable; for now, the three `examples/*.py` scripts are the smoke test — if they run, the kernel is healthy.)

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'rf_compute'` after `pip install -e .` | Install didn't complete, or you're in the wrong directory | `cd` to the repo root (where `pyproject.toml` lives) and re-run `pip install -e .` |
| `ImportError: No module named 'SoapySDR'` when using `backend="sdr"` | SoapySDR Python binding not installed | `pip install -e ".[sdr]"` — the extra installs `SoapySDR` |
| `SoapySDR.Device.enumerate()` returns `[]` | Drivers installed, no SDR plugged in | Plug in your HackRF or RTL-SDR |
| `SoapySDR.Device.enumerate()` raises `RuntimeError: no devices found` | SDR plugged in, drivers not found | Install the vendor driver (PothosSDR on Windows; `brew` on macOS; `apt` on Linux). See the OS-specific instructions above. |
| Examples run but `backend: sim (degraded: True)` | SDR backend requested but SoapySDR not on path | Check that the SoapySDR Python binding is importable: `python -c "import SoapySDR"`. If that fails, the system library isn't on the path. |
| `Permission denied` on Linux when accessing SDR | udev rules not installed | `sudo usermod -aG plugdev $USER` then log out and back in. HackRF and RTL-SDR need `plugdev` group membership on Linux. |
| HackRF shows up but `enumerate()` lists it as `hackrf` with no serial | Driver partial install | Reinstall HackRF drivers (PothosSDR on Windows). The serial number should appear in the enumerate output. |

---

## What to do next

Once you can run the examples, work through the walkthroughs in order:

1. [`docs/hello-world-aircomp.md`](hello-world-aircomp.md) — Tier 1, AirComp sum (needs 2× HackRF + 1× RTL-SDR, ~$350)
2. [`docs/hello-world-convolution.md`](hello-world-convolution.md) — Tier 2, wave-domain convolution (needs 1× HackRF + 1× RTL-SDR, ~$200)
3. [`docs/hello-world-matrix-inversion.md`](hello-world-matrix-inversion.md) — Tier 3, matrix inversion (3 modes: free / $200 / $350)

Each walkthrough is self-contained: hardware list, parts costs, code, expected output, troubleshooting, safety.

If you want the theoretical context first, read [`docs/bibliography.md`](bibliography.md) — it has a four-phase reading order (~8–10 hours) that takes you from the core insight to the frontier.

If you want to contribute, read [`CONTRIBUTING.md`](../CONTRIBUTING.md) — it defines the reproducibility, provenance, and honesty bars, and names the open frontier this surface points at.

---

## Provenance

- **Package:** `rf-compute` — a research and education surface for wave-domain computation
- **License:** MIT (see `LICENSE`)
- **Python support:** 3.9, 3.10, 3.11, 3.12
- **Hardware support:** HackRF One (Tx), RTL-SDR (Rx) — via SoapySDR with graceful degradation to simulation
- **Install method:** PEP 621 pyproject.toml; `pip install -e .` for development, `pip install .` for production