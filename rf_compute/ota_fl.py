"""
Over-the-air federated learning — the Tier 4 capstone
=====================================================

Tiers 1-1.6 built the primitive: the channel adds. Tier 4 puts that adder
to work on the job it is most often proposed for — aggregating model
updates in federated learning. Every round, K devices compute a gradient on
their own data and transmit it AT THE SAME TIME; the channel superposes the
transmissions, and the server reads the sum. One aggregation costs d+1
channel uses (plus a short pilot) no matter how many devices take part.
Orthogonal transmission (each device in its own slot) costs K*d.

The lineage: Zhu, Wang & Huang, "Broadband Analog Aggregation for
Low-Latency Federated Edge Learning," IEEE TWC 19(1) 491-506 (2020),
DOI 10.1109/TWC.2019.2946245; Yang, Jiang, Shi & Ding, "Federated Learning
via Over-the-Air Computation," IEEE TWC 19(3) 2022-2035 (2020),
DOI 10.1109/TWC.2019.2961673; Amiri & Gunduz, "Machine Learning at the
Wireless Edge: Distributed Stochastic Gradient Descent Over-the-Air," IEEE
TSP 68 2155-2169 (2020), DOI 10.1109/TSP.2020.2981904.

What goes wrong on real hardware: misalignment
----------------------------------------------
Tier 1's parts list includes a 10 MHz clock cable for a reason. AirComp
assumes every transmission arrives at the same instant with the same phase.
Real devices miss both:

  - PHASE: channel-inversion pre-coding removes the channel phase only as
    well as the channel estimate allows. Device k arrives rotated by a
    residual phi_k.
  - TIMING: device k's symbols arrive tau_k of a symbol late. With
    rectangular pulses, the receiver's sample j then holds (1 - tau_k) of
    symbol j and tau_k of symbol j-1 — inter-symbol interference.

The received samples become

    r_j = sqrt(eta) * sum_k e^{i phi_k} [(1 - tau_k) g_k[j] + tau_k g_k[j-1]] + z_j

Misalignment model: Shao, Gunduz & Liew, "Federated Edge Learning With
Misaligned Over-the-Air Computation," IEEE TWC 21(6) 3951-3964 (2022),
DOI 10.1109/TWC.2021.3125798; official code (MIT): hku-icl/MisAlignedOAC.

Two receivers
-------------
  naive      — takes Re(r_j) as the sum, as if everything were aligned.
  equalized  — every device sends the same known pilot simultaneously. The
               pilot measures the AGGREGATE taps
                   c0 = sum_k e^{i phi_k} (1 - tau_k),   c1 = sum_k e^{i phi_k} tau_k
               — never any single device's offset. The receiver then solves
               the two-tap deconvolution r/sqrt(eta) = c0 u[j] + c1 u[j-1]
               for a real vector u by least squares.

If every device sent the same gradient, u would be that gradient exactly.
They don't, so the equalizer is exact for the COMMON part of the gradients
and leaves the device-to-device differences smeared by the offsets. That
is the lesson this tier measures: misalignment error scales with how
different the devices' data are (heterogeneity), not just with the size
of the offsets. A shared clock shrinks the offsets; nothing at a one-sample
receiver removes the heterogeneity term.

Scope (toy): one sample per symbol, known pulse shape, real-valued model.
Shao et al. oversample with a whitened matched filter and decode with a
sum-product ML estimator or an aligned-sample estimator — strictly stronger
receivers. The equalizer here is the one-screen version that shows why a
receiver needs to know about misalignment at all.

Module map
----------
    make_federated_data(...)  — K devices of logistic-regression data, tunable non-IID
    local_gradient(w, X, y)   — the logistic-loss gradient one device computes
    ota_aggregate(...)        — one over-the-air aggregation round (the wave domain)
    train(...)                — federated training under one aggregation scheme
    scoreline(...)            — every scheme on the same data, same seed

All randomness flows through a seeded numpy Generator. Same seed, same numbers.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    'SCHEMES', 'make_federated_data', 'local_gradient', 'gradient_spread',
    'ota_aggregate', 'aggregation_quality', 'train', 'scoreline',
]

# name -> (over the air?, misaligned?, receiver)
SCHEMES = {
    'orthogonal':          (False, False, None),
    'ota_aligned':         (True, False, 'naive'),
    'ota_misaligned':      (True, True, 'naive'),
    'ota_equalized':       (True, True, 'equalized'),
}


# ─────────────────────────────────────────────────────────────────────────────
# The learning problem
# ─────────────────────────────────────────────────────────────────────────────

def make_federated_data(num_devices=20, samples_per_device=200, dim=20,
                        heterogeneity=1.0, test_samples=2000, rng=None):
    """
    Binary logistic-regression data split across devices.

    One ground-truth weight vector labels everything. Device k's features are
    drawn around its own center mu_k = heterogeneity * (random unit vector),
    so at heterogeneity 0 every device sees the same distribution (IID) and
    as it grows the devices' local gradients point in increasingly different
    directions (non-IID). The test set is drawn from the mixture of all
    devices' distributions.

    Returns (devices, (X_test, y_test), w_true), with devices a list of (X_k, y_k).
    """
    if rng is None:
        rng = np.random.default_rng()
    w_true = rng.normal(0.0, 1.0, size=dim)
    w_true *= 3.0 / np.linalg.norm(w_true)

    def draw(center, m):
        X = rng.normal(0.0, 1.0, size=(m, dim)) + center
        p = 1.0 / (1.0 + np.exp(-X @ w_true))
        y = (rng.uniform(size=m) < p).astype(np.float64)
        return X, y

    centers = []
    devices = []
    for _ in range(num_devices):
        u = rng.normal(0.0, 1.0, size=dim)
        center = heterogeneity * u / np.linalg.norm(u)
        centers.append(center)
        devices.append(draw(center, samples_per_device))

    owners = rng.integers(0, num_devices, size=test_samples)
    X_test = rng.normal(0.0, 1.0, size=(test_samples, dim)) + np.asarray(centers)[owners]
    p = 1.0 / (1.0 + np.exp(-X_test @ w_true))
    y_test = (rng.uniform(size=test_samples) < p).astype(np.float64)
    return devices, (X_test, y_test), w_true


def local_gradient(w, X, y, l2=1e-3):
    """Gradient of the mean logistic loss (plus a small L2 term) on one device's data."""
    p = 1.0 / (1.0 + np.exp(-X @ w))
    return X.T @ (p - y) / len(y) + l2 * w


def gradient_spread(devices, w=None, clip=1.0):
    """
    How different the devices' gradients are at w (default: the zero start):
        mean_k ||g_k - g_bar||^2 / ||g_bar||^2
    0 means every device sends the same gradient. This, not the heterogeneity
    knob itself, is the quantity the equalizer's residual error scales with.
    """
    d = devices[0][0].shape[1]
    w = np.zeros(d) if w is None else w
    G = np.array([_clip(local_gradient(w, X, y), clip) for X, y in devices])
    g_bar = G.mean(axis=0)
    return float(np.mean(np.sum((G - g_bar) ** 2, axis=1)) / np.sum(g_bar ** 2))


def _loss_and_accuracy(w, X, y):
    z = X @ w
    loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
    acc = float(np.mean((z > 0) == (y > 0.5)))
    return loss, acc


# ─────────────────────────────────────────────────────────────────────────────
# One over-the-air aggregation — the wave domain
# ─────────────────────────────────────────────────────────────────────────────

def _clip(g, c):
    n = float(np.linalg.norm(g))
    return g if n <= c else g * (c / n)


def ota_aggregate(grads, snr_db=10.0, rng=None, *, truncation=0.3, clip=1.0,
                  max_phase=0.0, max_timing=0.0, receiver='equalized',
                  pilot_len=32):
    """
    Aggregate K gradient vectors over a shared fading channel.

    Per round (block fading):
      1. h_k ~ CN(0, 1). Devices with |h_k| < truncation sit this round out
         (truncated channel inversion — the deep-fade lesson of Tier 1.6).
      2. Each participant clips its gradient to norm <= clip and transmits
         x_k = sqrt(eta) * g_k * conj(h_k)/|h_k|^2, so the channel delivers
         sqrt(eta) * g_k. eta = P * d * min_k |h_k|^2 / clip^2 (over the
         participants) is the largest common scaling that keeps every
         participant within its per-symbol power budget P: the weakest
         channel sets the scale for everyone.
      3. Misalignment (optional): device k arrives with residual phase
         phi_k ~ U(-max_phase, max_phase) and timing offset
         tau_k ~ U(0, max_timing) symbols.
      4. The receiver sees the superposition plus CN(0, sigma^2) noise,
         sigma^2 = P / SNR, over d+1 sample windows, and estimates the
         AVERAGE gradient of the participants.

    receiver: 'naive' (Re(r) / sqrt(eta) / K_active) or 'equalized'
    (pilot-aided two-tap least squares; see the module docstring).

    Returns a dict: 'estimate', 'true_average' (of the participants' clipped
    gradients, for measuring the aggregation error), 'mse', 'nmse' (mse
    normalized by the true average's mean square — scale-free, so it stays
    comparable as gradients shrink during training), 'participants',
    'max_tx_power' (the largest per-symbol transmit power any participant
    used; never above the budget P = 1), 'channel_uses' (d + 1 + pilot_len).
    """
    if rng is None:
        rng = np.random.default_rng()
    if receiver not in ('naive', 'equalized'):
        raise ValueError(f"receiver '{receiver}' not supported; use 'naive' or 'equalized'")
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
                'channel_uses': d + 1 + pilot_len}
    G = np.array([_clip(grads[k], clip) for k in active])      # (K_active, d)
    true_avg = G.mean(axis=0)
    h_min2 = float(np.min(np.abs(h[active]) ** 2))
    eta = P * d * h_min2 / clip ** 2
    # Per-symbol power of x_k = sqrt(eta) g_k conj(h_k)/|h_k|^2 is
    # eta ||g_k||^2 / (d |h_k|^2) <= eta clip^2 / (d h_min2) = P.
    max_tx_power = float(np.max(eta * np.sum(G ** 2, axis=1) / (d * np.abs(h[active]) ** 2)))

    phi = rng.uniform(-max_phase, max_phase, size=len(active))
    tau = rng.uniform(0.0, max_timing, size=len(active))
    rot = np.exp(1j * phi)

    def noise(m):
        return np.sqrt(sigma2 / 2.0) * (rng.normal(size=m) + 1j * rng.normal(size=m))

    # The superposition over d+1 windows: sample j holds (1-tau) of symbol j
    # and tau of symbol j-1 (symbol -1 and symbol d are silence).
    padded = np.zeros((len(active), d + 1))
    padded[:, :d] = G
    prev = np.zeros_like(padded)
    prev[:, 1:] = padded[:, :-1]
    per_device = (1.0 - tau)[:, None] * padded + tau[:, None] * prev
    r = np.sqrt(eta) * (rot[:, None] * per_device).sum(axis=0) + noise(d + 1)

    if receiver == 'naive':
        estimate = r[:d].real / np.sqrt(eta) / len(active)
    else:
        # Pilot: every participant sends the same +/-1 sequence at the same
        # time and the same scaling. The receiver fits the two aggregate taps.
        p = rng.choice([-1.0, 1.0], size=pilot_len)
        p_prev = np.concatenate(([0.0], p[:-1]))
        c0 = np.sum(rot * (1.0 - tau))
        c1 = np.sum(rot * tau)
        eta_p = P * h_min2                   # a +/-1 pilot has per-symbol power 1
        rp = np.sqrt(eta_p) * (c0 * p + c1 * p_prev) + noise(pilot_len)
        A = np.column_stack([p, p_prev]) * np.sqrt(eta_p)
        c0_hat, c1_hat = np.linalg.lstsq(A, rp, rcond=None)[0]
        # Two-tap convolution matrix, (d+1) x d, solved for a REAL u.
        T = np.zeros((d + 1, d), dtype=np.complex128)
        idx = np.arange(d)
        T[idx, idx] = c0_hat
        T[idx + 1, idx] = c1_hat
        A_real = np.vstack([T.real, T.imag])
        b_real = np.concatenate([r.real, r.imag]) / np.sqrt(eta)
        estimate = np.linalg.lstsq(A_real, b_real, rcond=None)[0]

    mse = float(np.mean((estimate - true_avg) ** 2))
    power = float(np.mean(true_avg ** 2))
    return {
        'estimate': estimate,
        'true_average': true_avg,
        'mse': mse,
        'nmse': mse / power if power > 0 else 0.0,
        'participants': int(len(active)),
        'max_tx_power': max_tx_power,
        'channel_uses': d + 1 + pilot_len,
    }


def aggregation_quality(devices, snr_db=20.0, max_phase=np.pi, max_timing=1.0,
                        trials=400, seed=0, truncation=0.3, clip=1.0):
    """
    How well one aggregation recovers the average gradient, measured at a
    FIXED point (the zero start, where gradients are largest) over many
    independent channel draws. Returns the median NMSE for the aligned
    channel, the misaligned channel with the naive receiver, and the
    misaligned channel with the equalizer.

    Measured at a fixed point on purpose: during training the gradients
    shrink toward zero while the channel noise does not, so a per-round
    NMSE averaged over a run mostly measures how close training got to
    convergence.
    """
    rng = np.random.default_rng(seed)
    d = devices[0][0].shape[1]
    grads = [local_gradient(np.zeros(d), X, y) for X, y in devices]
    cases = {
        'ota_aligned': (0.0, 0.0, 'naive'),
        'ota_misaligned': (max_phase, max_timing, 'naive'),
        'ota_equalized': (max_phase, max_timing, 'equalized'),
    }
    out = {}
    for name, (ph, tm, rx) in cases.items():
        nmse = [ota_aggregate(grads, snr_db=snr_db, rng=rng, truncation=truncation,
                              clip=clip, max_phase=ph, max_timing=tm,
                              receiver=rx)['nmse'] for _ in range(trials)]
        out[name] = float(np.median(nmse))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Federated training
# ─────────────────────────────────────────────────────────────────────────────

def train(scheme='ota_equalized', devices=None, test=None, rounds=60, lr=0.5,
          snr_db=10.0, truncation=0.3, clip=1.0, max_phase=np.pi / 3,
          max_timing=0.5, rng=None):
    """
    Federated gradient descent (one local gradient per device per round),
    aggregated by the chosen scheme:

      orthogonal      — every device in its own slot, received error-free:
                        the exact average of all K clipped gradients.
                        K*d channel uses per round. The upper bound.
      ota_aligned     — over-the-air, perfect sync, noisy, truncated.
      ota_misaligned  — over-the-air with phase and timing offsets, naive receiver.
      ota_equalized   — same offsets, pilot-aided equalizing receiver.

    Returns a dict of per-round histories: 'test_accuracy', 'test_loss',
    'aggregation_nmse' (0 for orthogonal), 'participants', and the total
    'channel_uses'.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme '{scheme}' not supported; use one of {sorted(SCHEMES)}")
    if rng is None:
        rng = np.random.default_rng()
    if devices is None or test is None:
        devices, test, _ = make_federated_data(rng=rng)
    over_air, misaligned, receiver = SCHEMES[scheme]
    X_test, y_test = test
    d = devices[0][0].shape[1]
    w = np.zeros(d)
    hist = {'test_accuracy': [], 'test_loss': [], 'aggregation_nmse': [],
            'participants': [], 'channel_uses': 0}
    for _ in range(rounds):
        grads = [local_gradient(w, X, y) for X, y in devices]
        if over_air:
            out = ota_aggregate(
                grads, snr_db=snr_db, rng=rng, truncation=truncation, clip=clip,
                max_phase=max_phase if misaligned else 0.0,
                max_timing=max_timing if misaligned else 0.0,
                receiver=receiver)
            step = out['estimate']
            hist['aggregation_nmse'].append(out['nmse'])
            hist['participants'].append(out['participants'])
            hist['channel_uses'] += out['channel_uses']
        else:
            step = np.mean([_clip(g, clip) for g in grads], axis=0)
            hist['aggregation_nmse'].append(0.0)
            hist['participants'].append(len(devices))
            hist['channel_uses'] += len(devices) * d
        w = w - lr * step
        loss, acc = _loss_and_accuracy(w, X_test, y_test)
        hist['test_loss'].append(loss)
        hist['test_accuracy'].append(acc)
    hist['weights'] = w
    return hist


def scoreline(num_devices=20, dim=20, samples_per_device=200, heterogeneity=2.0,
              rounds=60, lr=0.5, snr_db=20.0, max_phase=np.pi,
              max_timing=1.0, seed=7, num_seeds=10):
    """
    The Tier 4 experiment. For each of num_seeds seeds (seed, seed+1, ...),
    draw a fresh federated dataset and train every scheme on it with the
    same channel randomness. Reports, per scheme:

      accuracy_mean / accuracy_min / accuracy_max — final test accuracy
                        across seeds. Report the spread, not one run: with
                        no phase sync the naive receiver's final accuracy
                        is a coin flip (one lucky seed looks fine).
      loss_mean       — final test loss, averaged over seeds.
      aggregation_nmse — fixed-point aggregation NMSE (aggregation_quality),
                        averaged over seeds.
      mean_participants, channel_uses — per run.

    Plus the data's gradient_spread (averaged over seeds).

    The defaults are the severe case: no phase synchronization at all
    (phi ~ U(-pi, pi)) and up to one symbol of timing offset — hardware
    without the shared clock.
    """
    runs = {k: {'acc': [], 'loss': [], 'nmse': [], 'part': [], 'uses': 0} for k in SCHEMES}
    spreads = []
    for s in range(seed, seed + num_seeds):
        devices, test, _ = make_federated_data(
            num_devices=num_devices, samples_per_device=samples_per_device,
            dim=dim, heterogeneity=heterogeneity, rng=np.random.default_rng(s))
        spreads.append(gradient_spread(devices))
        quality = aggregation_quality(devices, snr_db=snr_db, max_phase=max_phase,
                                      max_timing=max_timing, seed=s)
        for scheme in SCHEMES:
            h = train(scheme, devices=devices, test=test, rounds=rounds, lr=lr,
                      snr_db=snr_db, max_phase=max_phase, max_timing=max_timing,
                      rng=np.random.default_rng(s + 10_000))
            r = runs[scheme]
            r['acc'].append(h['test_accuracy'][-1])
            r['loss'].append(h['test_loss'][-1])
            r['nmse'].append(quality.get(scheme, 0.0))
            r['part'].append(float(np.mean(h['participants'])))
            r['uses'] = h['channel_uses']
    out = {'heterogeneity': heterogeneity, 'snr_db': snr_db, 'rounds': rounds,
           'num_devices': num_devices, 'max_phase': max_phase,
           'max_timing': max_timing, 'num_seeds': num_seeds,
           'gradient_spread': float(np.mean(spreads))}
    for scheme, r in runs.items():
        out[scheme] = {
            'accuracy_mean': float(np.mean(r['acc'])),
            'accuracy_min': float(np.min(r['acc'])),
            'accuracy_max': float(np.max(r['acc'])),
            'loss_mean': float(np.mean(r['loss'])),
            'aggregation_nmse': float(np.mean(r['nmse'])),
            'mean_participants': float(np.mean(r['part'])),
            'channel_uses': r['uses'],
        }
    return out
