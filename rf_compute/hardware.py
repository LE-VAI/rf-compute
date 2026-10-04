"""
Hardware reality layer — what the idealized channel model leaves out
====================================================================

Tier 4 models misalignment as two numbers per device: a residual phase
phi_k and a timing offset tau_k, both drawn once per round and held CONSTANT
across the whole block. That is the right first model, and it is the model
Shao, Gunduz & Liew use. But it is not what a $2-$25 board does.

A real ESP32-class receiver (ESP-SDR / eSpDR / ESP-PPB class) adds
impairments the simulator has no basis function for:

    cfo          a phase RAMP across the block — the LO is not calibrated, so
                 the residual offset does not sit still, it accumulates.
    phase noise  a random WALK on top of the ramp. This is the impairment that
                 made eSpDR's first raw-I/Q streaming unusable until the
                 clocking was fixed (I2S master-clock forwarding).
    sfo          a sampling-rate offset that makes tau_k DRIFT across the
                 block, so the two aggregate taps are not constants — they are
                 functions of the sample index.
    iq imbalance a receiver-side image spur; the ADC path is uncalibrated.
    quantization signed 10-bit I/Q (the format the disclosure documents), not
                 float64.
    agc          an unknown residual gain error, so the receiver's assumed eta
                 is slightly wrong.
    duty cycle   ESP32-class capture is BURST, not streaming. Samples between
                 capture windows are simply missing. The disclosure says it
                 plainly: "packets that arrive between capture windows may be
                 missed."

Why this matters for the equalizer
----------------------------------
Tier 4's equalizer measures the aggregate taps from a pilot and applies them
to the whole data block. That is exact when the taps are constant. Under a
phase ramp and an SFO the taps are TIME-VARYING, so the measurement is stale
the moment it is taken — and the error is a function of position within the
block, not merely of the offsets' size. The simulator cannot show this,
because a constant-per-round phi_k has no drift to be stale about.

So this module answers a question with a number instead of an argument: does
the Tier 4 equalizer survive a real receiver, or does the bridge need a
tracking loop? And if it needs one, what does the fix cost?

Lineage
-------
The impairment models follow standard SDR practice: carrier frequency offset
and phase noise as a two-part process (ramp plus Wiener walk), sampling
frequency offset as a symbol-rate drift, IQ imbalance as a complex image gain,
and ADC quantization as uniform rounding at the documented bit depth. Periodic
in-block pilots (midambles) as the remedy are the standard cellular
construction (LTE/NR DMRS). The ESP32 capture constraints — signed 10-bit I/Q,
burst duty cycle — are from the ESP-SDR disclosure at espargos.net/espsdr/
(2026-09-30) together with the follow-on work by eSpDR, SoapyESPSDR and
ESP-PPB.

Scope
-----
Toy-scaled and deliberately simple, like the rest of the package: one sample
per symbol, real-valued gradient symbols, no pulse shaping. Every function is
seeded and reproducible. Nothing here transmits; this is a receiver-side and
channel-side model only.

The impairment PRESETS are NOT measurements. They are stated assumptions
chosen to bracket the plausible range, and every number in them should be
replaced with a bench measurement as soon as one exists. They are labelled
that way on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .ota_fl import _clip, local_gradient

__all__ = [
    'HardwareImpairments',
    'PHONE_GRADE', 'DEV_BOARD_UNCALIBRATED', 'DEV_BOARD_CLOCKED',
    'esp_iq_word_layout', 'unpack_esp_iq', 'impair_received',
    'ota_aggregate_hardware', 'survival_table', 'overhead_table',
]


# ─────────────────────────────────────────────────────────────────────────────
# The impairment set
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HardwareImpairments:
    """
    One receiver/channel impairment configuration.

    Every field defaults to "ideal", so `HardwareImpairments()` reproduces the
    Tier 4 model exactly and any single field can be switched on alone. That is
    deliberate: an impairment layer you cannot turn off cannot tell you which
    impairment did the damage.

    Fields, with the hardware they stand for:

      cfo_rad_per_symbol      Residual carrier offset as phase accumulation PER
                              SYMBOL, drawn INDEPENDENTLY per device from
                              U(-cfo, +cfo). This is the spread that an
                              unsynchronised set of nodes has, and it is the
                              impairment this module was built to measure.
      cfo_common_rad_per_symbol
                              Carrier offset SHARED by every device, drawn once
                              per transmission. This is what a common clock
                              leaves behind: one rotation rate everyone agrees
                              on. The distinction between this field and the one
                              above is the whole argument for synchronization,
                              and `sync_isolation_table` measures it.
      phase_noise_rad         Per-symbol innovation std of a Wiener phase walk on
                              top of the ramp.
      sfo_symbols_per_block   Total symbol-timing drift across one transmission,
                              per device, drawn once. Turns tau_k from a constant
                              into a ramp — this is what makes taps go stale.
      iq_imbalance_db         Image-suppression figure. -100 dB is ideal.
      adc_bits                Quantization depth. 0 disables. The documented
                              ESP32 raw-I/Q path is signed 10-bit.
      agc_gain_error_db       Residual error in the receiver's assumed scaling
                              eta, as log-normal spread in dB, drawn per block.
      duty_cycle              Fraction of sample windows captured. 1.0 is
                              continuous; ESP32-class capture is burst, so below
                              1.0 the missing windows carry NaN and the receiver
                              sees nothing there.
      gap_model               'scatter' (windows dropped at random) or 'burst'
                              (a contiguous stretch dropped).
    """
    cfo_rad_per_symbol: float = 0.0
    cfo_common_rad_per_symbol: float = 0.0
    phase_noise_rad: float = 0.0
    sfo_symbols_per_block: float = 0.0
    iq_imbalance_db: float = -100.0
    adc_bits: int = 0
    agc_gain_error_db: float = 0.0
    duty_cycle: float = 1.0
    gap_model: str = 'burst'

    def __post_init__(self):
        if not 0.0 < self.duty_cycle <= 1.0:
            raise ValueError("duty_cycle must be in (0, 1]")
        if self.adc_bits < 0:
            raise ValueError("adc_bits must be >= 0 (0 disables quantization)")
        if self.gap_model not in ('scatter', 'burst'):
            raise ValueError("gap_model must be 'scatter' or 'burst'")

    @property
    def is_ideal(self) -> bool:
        """True when every impairment is off — i.e. exactly the Tier 4 model."""
        return (self.cfo_rad_per_symbol == 0.0 and self.cfo_common_rad_per_symbol == 0.0
                and self.phase_noise_rad == 0.0
                and self.sfo_symbols_per_block == 0.0 and self.iq_imbalance_db <= -99.0
                and self.adc_bits == 0 and self.agc_gain_error_db == 0.0
                and self.duty_cycle == 1.0)

    def as_dict(self) -> dict:
        return asdict(self)


# Presets let an experiment name its hardware instead of its parameters.
# These are ASSUMPTIONS, not measurements. Replace with bench data.
#
# The three presets bracket the plausible range: a front end that is already
# synchronised, a dev board whose nodes have been brought into sync by an
# over-the-air stage, and a dev board with no sync at all.

PHONE_GRADE = HardwareImpairments(
    cfo_rad_per_symbol=0.02,
    phase_noise_rad=0.002,
    sfo_symbols_per_block=0.01,
    iq_imbalance_db=-45.0,
    adc_bits=12,
    agc_gain_error_db=0.1,
    duty_cycle=1.0,
)

DEV_BOARD_UNCALIBRATED = HardwareImpairments(
    cfo_rad_per_symbol=0.35,
    phase_noise_rad=0.02,
    sfo_symbols_per_block=0.25,
    iq_imbalance_db=-30.0,
    adc_bits=10,
    agc_gain_error_db=1.0,
    duty_cycle=0.6,
    gap_model='burst',
)

DEV_BOARD_CLOCKED = HardwareImpairments(
    cfo_rad_per_symbol=0.05,
    phase_noise_rad=0.005,
    sfo_symbols_per_block=0.02,
    iq_imbalance_db=-35.0,
    adc_bits=10,
    agc_gain_error_db=0.3,
    duty_cycle=0.9,
    gap_model='scatter',
)


# ─────────────────────────────────────────────────────────────────────────────
# The capture format the hardware actually produces
# ─────────────────────────────────────────────────────────────────────────────

def esp_iq_word_layout() -> dict:
    """
    The documented ESP32 raw-I/Q word layout, as a shift/mask table.

    From the ESP-SDR disclosure: each sample is one 32-bit word carrying signed
    10-bit I and Q alongside gain and AGC state. Returning this as data, rather
    than burying it in the unpacker, makes the format auditable and lets a test
    pin it.
    """
    return {
        'word_bits': 32,
        'i':       {'shift': 10, 'bits': 10, 'signed': True},
        'q':       {'shift': 0,  'bits': 10, 'signed': True},
        'gain':    {'shift': 20, 'bits': 8,  'signed': False},
        'agc_fsm': {'shift': 28, 'bits': 4,  'signed': False},
    }


def _sign_extend(value: np.ndarray, bits: int) -> np.ndarray:
    """Two's-complement sign extension for a `bits`-wide field."""
    sign_bit = np.int64(1) << np.int64(bits - 1)
    mask = (np.int64(1) << np.int64(bits)) - np.int64(1)
    v = value & mask
    return np.where(v >= sign_bit, v - (np.int64(1) << np.int64(bits)), v)


def pack_esp_iq(i, q, *, gain=0, agc_fsm=0) -> np.ndarray:
    """
    Pack integer I/Q (and optional gain / AGC state) into 32-bit words.

    The inverse of `unpack_esp_iq`. Exists mainly so the round trip can be
    tested, and so a synthetic capture can be written in the real format.
    """
    L = esp_iq_word_layout()
    i = np.asarray(i, dtype=np.int64) & np.int64((1 << L['i']['bits']) - 1)
    q = np.asarray(q, dtype=np.int64) & np.int64((1 << L['q']['bits']) - 1)
    g = np.asarray(gain, dtype=np.int64) & np.int64((1 << L['gain']['bits']) - 1)
    a = np.asarray(agc_fsm, dtype=np.int64) & np.int64((1 << L['agc_fsm']['bits']) - 1)
    return ((i << np.int64(L['i']['shift']))
            | (q << np.int64(L['q']['shift']))
            | (g << np.int64(L['gain']['shift']))
            | (a << np.int64(L['agc_fsm']['shift'])))


def unpack_esp_iq(words, *, scale: Optional[float] = None) -> np.ndarray:
    """
    Unpack ESP32 raw-I/Q 32-bit words into a complex sample vector.

    This is the first thing anyone with a board in hand has to write, and the
    bit layout is easy to get subtly wrong — hence a pinned implementation.

    Parameters
    ----------
    words : array-like of uint32/int64
        One 32-bit word per sample.
    scale : float, optional
        Divisor applied to the integer I/Q. Default None returns the raw signed
        integers (10-bit range, -512..511). Pass a float to normalise, e.g.
        512.0 for nominal full scale.

    Returns
    -------
    complex128 array, I + 1j*Q.

    Notes
    -----
    Gain and AGC-FSM fields are decoded by `esp_iq_word_layout()` but NOT
    returned here: the disclosure documents the gain field's position, not its
    unit, so applying it as a calibration would be inventing a scale factor.
    Gain handling is deliberately left to the caller.
    """
    w = np.asarray(words, dtype=np.int64).ravel()
    L = esp_iq_word_layout()
    i = _sign_extend(w >> np.int64(L['i']['shift']), L['i']['bits'])
    q = _sign_extend(w >> np.int64(L['q']['shift']), L['q']['bits'])
    out = i.astype(np.float64) + 1j * q.astype(np.float64)
    if scale is not None:
        out = out / float(scale)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Receiver-side impairments on the aggregate
# ─────────────────────────────────────────────────────────────────────────────

def _quantize(x: np.ndarray, bits: int) -> np.ndarray:
    """Uniform mid-tread quantization of a complex signal, `bits` per rail."""
    if bits <= 0:
        return x
    levels = float((1 << (bits - 1)) - 1)
    peak = float(np.max(np.abs(x)))
    if peak == 0.0:
        return x
    return np.round(x / peak * levels) / levels * peak


def impair_received(r: np.ndarray, *, rng, cfg: HardwareImpairments) -> np.ndarray:
    """
    Apply the RECEIVER-side half of the impairment chain to a received block.

    These act on the AGGREGATE signal, after the channel has already summed the
    devices, so they are applied once rather than per device:

        IQ imbalance  -> image spur
        AGC           -> unknown scale on the whole block
        ADC           -> quantization at the documented depth
        duty cycle    -> missing capture windows, filled with NaN

    The per-device impairments (each device's own CFO, phase noise, SFO) are
    applied per device INSIDE the aggregation, because they act before the sum
    and cannot be commuted through it.

    Returns a new array; the input is not modified.
    """
    out = np.asarray(r, dtype=np.complex128).copy()

    # IQ imbalance: a conjugate image sitting X dB below the signal.
    g = 10.0 ** (cfg.iq_imbalance_db / 20.0)
    if g > 0.0:
        out = out + g * np.conj(out)

    # AGC: the receiver's assumed eta is off. Modelled as a scale, because that
    # is exactly what a gain error does to a channel-inversion scheme — it breaks
    # the KNOWN-NESS of the scale, not the linearity.
    if cfg.agc_gain_error_db != 0.0:
        out = out * (10.0 ** (rng.normal(0.0, cfg.agc_gain_error_db) / 20.0))

    # ADC depth.
    if cfg.adc_bits > 0:
        out = _quantize(out, cfg.adc_bits)

    # Burst capture: the windows that were never captured.
    if cfg.duty_cycle < 1.0:
        n = out.size
        n_missing = int(round(n * (1.0 - cfg.duty_cycle)))
        if n_missing > 0:
            if cfg.gap_model == 'burst':
                start = int(rng.integers(0, max(1, n - n_missing + 1)))
                out[start:start + n_missing] = np.nan
            else:
                idx = rng.choice(n, size=n_missing, replace=False)
                out[idx] = np.nan
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-device timelines and superposition
# ─────────────────────────────────────────────────────────────────────────────

def _timelines(n_active: int, length: int, *, rng, cfg: HardwareImpairments,
               max_phase: float, max_timing: float):
    """
    Each device's own timing and phase trajectory across a transmission of
    `length` samples. These are the impairments that must be applied BEFORE the
    superposition, because the channel sums signals, not impairments.

    Returns (tau, theta), each (n_active, length):

        tau_k(j)   = tau_k(0) + sfo_k * j / length      -> sampling drift
        theta_k(j) = phi_k + cfo_k * j + walk_k(j)      -> LO ramp + phase noise
    """
    idx = np.arange(length, dtype=np.float64)
    tau0 = rng.uniform(0.0, max_timing, size=n_active)
    cfo_common = rng.uniform(-cfg.cfo_common_rad_per_symbol,
                             cfg.cfo_common_rad_per_symbol)
    cfo = (cfo_common
           + rng.uniform(-cfg.cfo_rad_per_symbol, cfg.cfo_rad_per_symbol, size=n_active))
    sfo = rng.uniform(-cfg.sfo_symbols_per_block, cfg.sfo_symbols_per_block, size=n_active)
    phi0 = rng.uniform(-max_phase, max_phase, size=n_active)

    tau = tau0[:, None] + (sfo[:, None] / float(max(length, 1))) * idx[None, :]
    theta = phi0[:, None] + cfo[:, None] * idx[None, :]

    if cfg.phase_noise_rad > 0.0 and length > 1:
        walk = np.zeros((n_active, length))
        walk[:, 1:] = np.cumsum(
            rng.normal(0.0, cfg.phase_noise_rad, size=(n_active, length - 1)), axis=1)
        theta = theta + walk
    return tau, theta


def _mix(seq: np.ndarray, tau: np.ndarray, theta: np.ndarray,
         prior: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply one device's timing drift and phase trajectory to its own transmitted
    symbol sequence.

    Sample j of that device's received stream holds (1 - tau_j) of symbol j and
    tau_j of symbol j-1 — the same one-sample-per-symbol ISI model Tier 4 uses,
    except that tau is now a function of j.

    `prior` is the value of the symbol transmitted immediately before this
    block, in THIS block's own units (length n_active). Without it the boundary
    would be modelled as silence, which is wrong for a mid-block midamble: a
    real transmitter sends pilot and data back to back, so the midamble's last
    symbol genuinely leaks into the first data window. Getting this right is
    what makes the midamble's contribution a KNOWN constant the receiver can
    subtract, rather than an unmodelled error.
    """
    prev = np.zeros_like(seq)
    prev[:, 1:] = seq[:, :-1]
    if prior is not None:
        prev[:, 0] = prior
    frac = np.clip(tau, 0.0, 1.0)
    return np.exp(1j * theta) * ((1.0 - frac) * seq + frac * prev)


# ─────────────────────────────────────────────────────────────────────────────
# Receivers
# ─────────────────────────────────────────────────────────────────────────────

def _fit_two_taps(rp, pilot, eta_p):
    """
    Least-squares fit of the two AGGREGATE taps from a known midamble.

    Window 0 is skipped because it is contaminated by whatever symbol preceded
    the midamble; windows 1..m-1 each hold c0*p[j] + c1*p[j-1] with both symbols
    known, so the fit needs no assumption about the boundary. The fitted c0, c1
    are dimensionless and carry over unchanged to the data, which is why the
    same pair works for both.
    """
    m = int(min(rp.size, pilot.size))
    if m < 2:
        return 1.0 + 0.0j, 0.0 + 0.0j
    r_use = rp[1:m]
    p_cur = pilot[1:m]
    p_prv = pilot[0:m - 1]
    A = np.column_stack([p_cur, p_prv]) * np.sqrt(eta_p)
    c = np.linalg.lstsq(A, r_use, rcond=None)[0]
    return complex(c[0]), complex(c[1])


def _solve_segment(r_seg, c0, c1, n, eta, prior_pilot_symbol):
    """
    Two-tap deconvolution for a REAL gradient segment of length n.

    Sample j of the segment holds, in the mean-gradient approximation,
    sqrt(eta) * (c0*u[j] + c1*u[j-1]), where u[-1] is the last symbol of the
    preceding midamble. That symbol is KNOWN (±1), so its contribution is a
    constant and moves to the right-hand side — which is what makes a mid-block
    midamble usable at all, since only n windows are then needed and nothing has
    to be assumed about what follows the segment.

    The subtraction is done in the data's own units: divide by sqrt(eta) FIRST,
    then remove c1 * p_last. The pilot is transmitted at its own scaling
    sqrt(eta_p), so correcting with the pilot's scaling instead would under- or
    over-subtract by sqrt(eta/eta_p) = clip / sqrt(d) — a silent, wrong-by-a-
    constant error that no test on the ideal model would catch.
    """
    if n <= 0:
        return np.zeros(0)
    r = np.asarray(r_seg)[:n]
    if r.size < n:
        r = np.concatenate([r, np.zeros(n - r.size)])
    if abs(c0) < 1e-12:
        return np.zeros(n)
    T = np.zeros((n, n), dtype=np.complex128)
    ii = np.arange(n)
    T[ii, ii] = c0
    if n > 1:
        T[ii[1:], ii[:-1]] = c1
    b = r / np.sqrt(eta)
    b[0] -= c1 * prior_pilot_symbol
    A_real = np.vstack([T.real, T.imag])
    b_real = np.concatenate([b.real, b.imag])
    return np.linalg.lstsq(A_real, b_real, rcond=None)[0]


# ─────────────────────────────────────────────────────────────────────────────
# The impaired aggregation
# ─────────────────────────────────────────────────────────────────────────────

def ota_aggregate_hardware(grads, *, cfg: HardwareImpairments, snr_db: float = 10.0,
                           rng=None, truncation: float = 0.3, clip: float = 1.0,
                           max_phase: float = 0.0, max_timing: float = 0.0,
                           receiver: str = 'equalized', pilot_len: int = 32,
                           sub_blocks: int = 1) -> dict:
    """
    One over-the-air aggregation round through a REAL receiver.

    Same structure and same return keys as `ota_fl.ota_aggregate` — same fading,
    same truncated channel inversion, same two receivers — with `cfg`'s
    impairments inserted where physics puts them:

        per device, before the sum : CFO ramp, phase-noise walk, sampling drift
        once, after the sum        : IQ image, AGC error, ADC depth, burst gaps

    `sub_blocks` is the receiver fix. At 1 the transmitter sends one pilot and
    then the whole gradient, and the receiver fits ONE pair of taps and holds it
    across the block — Tier 4's design, and the reason its taps go stale under
    drift. At n > 1 the transmission is divided into n data sub-blocks, each
    preceded by its own short midamble, so the receiver re-measures the taps close
    to where it uses them. That is the standard cellular construction (LTE/NR
    DMRS), and `overhead_table` prices what it costs.

    Returns the `ota_aggregate` dict plus:
        'coverage'     fraction of sample windows actually captured
        'nan_windows'  count of missing windows
        'overhead'     pilot channel uses per data channel use
    """
    if rng is None:
        rng = np.random.default_rng()
    if receiver not in ('naive', 'equalized'):
        raise ValueError(f"receiver '{receiver}' not supported; use 'naive' or 'equalized'")
    if sub_blocks < 1:
        raise ValueError("sub_blocks must be >= 1")

    grads = [np.asarray(g, dtype=np.float64) for g in grads]
    K = len(grads)
    d = len(grads[0])
    P = 1.0
    sigma2 = P / (10 ** (snr_db / 10))

    h = (rng.normal(size=K) + 1j * rng.normal(size=K)) / np.sqrt(2.0)
    active = np.flatnonzero(np.abs(h) >= truncation)
    if len(active) == 0:
        return {'estimate': np.zeros(d), 'true_average': np.zeros(d), 'mse': 0.0,
                'nmse': 1.0, 'participants': 0, 'max_tx_power': 0.0,
                'channel_uses': d, 'coverage': 1.0, 'nan_windows': 0, 'overhead': 0.0}

    G = np.array([_clip(grads[k], clip) for k in active])
    true_avg = G.mean(axis=0)
    h_min2 = float(np.min(np.abs(h[active]) ** 2))
    eta = P * d * h_min2 / clip ** 2
    max_tx_power = float(np.max(eta * np.sum(G ** 2, axis=1) / (d * np.abs(h[active]) ** 2)))
    eta_p = P * h_min2
    n_active = len(active)

    # ── frame layout: midamble -> data segment -> midamble -> data segment ...
    bounds = np.linspace(0, d, sub_blocks + 1).astype(int)
    seg_lens = [int(b - a) for a, b in zip(bounds[:-1], bounds[1:])]
    seg_lens = [s for s in seg_lens if s > 0] or [d]
    starts = np.concatenate([[0], np.cumsum(seg_lens)[:-1]]).astype(int)
    pilots = [rng.choice([-1.0, 1.0], size=pilot_len) for _ in seg_lens]

    # Assemble the frame as (kind, symbols, start, length). Pilot symbols are
    # unit power; data symbols are the clipped gradients. Both are scaled for
    # transmission after the per-device mixing, exactly as Tier 4 does. The
    # length always comes from the symbol array itself, so a midamble longer
    # than a data segment cannot desynchronise the cursor.
    frame = []
    for pilot, n, start in zip(pilots, seg_lens, starts):
        pilot_symbols = np.tile(pilot, (n_active, 1))
        frame.append(('pilot', pilot_symbols, int(start), int(pilot_symbols.shape[1])))
        data_symbols = G[:, start:start + n]
        frame.append(('data', data_symbols, int(start), int(data_symbols.shape[1])))

    total_len = int(sum(b[1].shape[1] for b in frame))
    tau, theta = _timelines(n_active, total_len, rng=rng, cfg=cfg,
                            max_phase=max_phase, max_timing=max_timing)

    # Superpose: mix each device's own stream by its own impairment, then sum.
    received = []
    cursor = 0
    prior_val = np.zeros(n_active)          # nothing precedes the first block
    for kind, sym, start, n in frame:
        mixed = _mix(sym, tau[:, cursor:cursor + n], theta[:, cursor:cursor + n],
                     prior=prior_val)
        scale = np.sqrt(eta_p) if kind == 'pilot' else np.sqrt(eta)
        received.append(scale * mixed.sum(axis=0))
        prior_val = sym[:, -1]              # this block's last symbol precedes the next
        cursor += n

    r_all = np.concatenate(received)
    r_all = r_all + np.sqrt(sigma2 / 2.0) * (rng.normal(size=total_len)
                                            + 1j * rng.normal(size=total_len))

    # Receiver-side impairments, once, after the sum.
    r_all = impair_received(r_all, rng=rng, cfg=cfg)
    nan_windows = int(np.sum(np.isnan(r_all)))
    coverage = float(np.mean(~np.isnan(r_all))) if r_all.size else 1.0
    # A missing window is a missing window: the receiver sees nothing there.
    r_all = np.nan_to_num(r_all, nan=0.0)

    # ── receivers
    #
    # One pass down the frame. The only state that carries is the most recent
    # fitted tap pair and the symbol that immediately preceded the current
    # segment — the midamble's last symbol, which is what leaks into the
    # segment's first window.
    estimate = np.zeros(d)
    cursor = 0
    taps = None
    prior_symbol = 0.0
    for kind, sym, start, n in frame:
        seg = r_all[cursor:cursor + n]
        if kind == 'pilot':
            if receiver == 'equalized':
                taps = _fit_two_taps(seg, sym[0, :], eta_p)
        else:
            if receiver == 'naive':
                estimate[start:start + n] = seg.real / np.sqrt(eta) / n_active
            else:
                c0, c1 = taps if taps is not None else (1.0 + 0j, 0.0 + 0j)
                estimate[start:start + n] = _solve_segment(
                    seg, c0, c1, n, eta, prior_symbol)
        prior_symbol = float(sym[0, -1])
        cursor += n

    mse = float(np.mean((estimate - true_avg) ** 2))
    power = float(np.mean(true_avg ** 2))
    overhead = float(sum(len(p) for p in pilots) / float(d))
    return {
        'estimate': estimate,
        'true_average': true_avg,
        'mse': mse,
        'nmse': mse / power if power > 0 else 0.0,
        'participants': int(n_active),
        'max_tx_power': max_tx_power,
        'channel_uses': total_len,
        'coverage': coverage,
        'nan_windows': nan_windows,
        'overhead': overhead,
    }


# ─────────────────────────────────────────────────────────────────────────────
# The experiments
# ─────────────────────────────────────────────────────────────────────────────

def _fixed_point_grads(devices, clip):
    d = devices[0][0].shape[1]
    return [local_gradient(np.zeros(d), X, y) for X, y in devices]


def survival_table(devices, *, configs=None, snr_db: float = 20.0,
                   max_phase=np.pi, max_timing: float = 0.5, trials: int = 200,
                   seed: int = 0, clip: float = 1.0, truncation: float = 0.3,
                   sub_blocks: int = 4) -> dict:
    """
    Cross the impairment configurations against the receivers, at a fixed point
    (the zero start, where gradients are largest), over many channel draws.
    Median NMSE per cell.

    Columns:
        aligned    ideal channel, naive receiver — the free lunch
        naive      impaired channel, naive receiver
        equalized  impaired channel, Tier 4's single-pilot-fit equalizer
        tracked    impaired channel, per-sub-block midamble re-fit

    A cell near 1.0 means the estimate carries no more information than
    returning zero. Read the 'ideal (Tier 4 model)' row as the published claim
    and every lower row as what hardware does to it.
    """
    from . import ota_fl

    if configs is None:
        configs = {
            'ideal (Tier 4 model)': HardwareImpairments(),
            'phone grade': PHONE_GRADE,
            'dev board, ota-synced': DEV_BOARD_CLOCKED,
            'dev board, uncalibrated': DEV_BOARD_UNCALIBRATED,
        }
    grads = _fixed_point_grads(devices, clip)
    rng = np.random.default_rng(seed)

    def _ideal(**kw):
        return float(np.median([
            ota_fl.ota_aggregate(grads, snr_db=snr_db, rng=rng, truncation=truncation,
                                 clip=clip, **kw)['nmse'] for _ in range(trials)]))

    out = {}
    for name, cfg in configs.items():
        cells = {'aligned': _ideal(max_phase=0.0, max_timing=0.0, receiver='naive')}
        if cfg.is_ideal:
            cells['naive'] = cells['aligned']
            cells['equalized'] = _ideal(max_phase=max_phase, max_timing=max_timing,
                                        receiver='equalized')
            cells['tracked'] = cells['equalized']
            cells['coverage'] = 1.0
        else:
            def _hw(**kw):
                vals, cov = [], []
                for _ in range(trials):
                    o = ota_aggregate_hardware(
                        grads, cfg=cfg, snr_db=snr_db, rng=rng, truncation=truncation,
                        clip=clip, max_phase=max_phase, max_timing=max_timing, **kw)
                    vals.append(o['nmse'])
                    cov.append(o['coverage'])
                return float(np.median(vals)), float(np.mean(cov))

            cells['naive'], cov_n = _hw(receiver='naive')
            cells['equalized'], _ = _hw(receiver='equalized', sub_blocks=1)
            cells['tracked'], _ = _hw(receiver='equalized', sub_blocks=sub_blocks)
            cells['coverage'] = cov_n
        out[name] = cells
    return out


def overhead_table(devices, *, cfg: HardwareImpairments = DEV_BOARD_UNCALIBRATED,
                   sub_block_counts=(1, 2, 4, 8), snr_db: float = 20.0,
                   max_phase=np.pi, max_timing: float = 0.5, trials: int = 200,
                   seed: int = 0, clip: float = 1.0, truncation: float = 0.3,
                   pilot_len: int = 32) -> dict:
    """
    Price the fix. More midambles cut the tracking error; they also cost channel
    uses. Returns, per sub-block count: median NMSE, pilot overhead as a fraction
    of the data block, and total channel uses per aggregation.

    The point is to make the trade a number rather than a preference — and to
    show where adding midambles stops paying for itself.
    """
    grads = _fixed_point_grads(devices, clip)
    rng = np.random.default_rng(seed)
    out = {}
    for n_sub in sub_block_counts:
        vals, covs, uses, ovh = [], [], [], []
        for _ in range(trials):
            o = ota_aggregate_hardware(
                grads, cfg=cfg, snr_db=snr_db, rng=rng, truncation=truncation,
                clip=clip, max_phase=max_phase, max_timing=max_timing,
                receiver='equalized', sub_blocks=n_sub, pilot_len=pilot_len)
            vals.append(o['nmse'])
            covs.append(o['coverage'])
            uses.append(o['channel_uses'])
            ovh.append(o['overhead'])
        out[int(n_sub)] = {
            'nmse': float(np.median(vals)),
            'coverage': float(np.mean(covs)),
            'channel_uses': int(np.mean(uses)),
            'pilot_overhead': float(np.mean(ovh)),
        }
    return out
