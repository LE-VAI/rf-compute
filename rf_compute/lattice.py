"""
Nested-lattice coding for AirComp — the Nazer/Gastpar construction
===================================================================

This module implements the lattice machinery behind "computation over
multiple-access channels" (Nazer & Gastpar, IEEE Trans. Inf. Theory 53(10),
3498-3516, 2007) and its namesake extension "Compute-and-forward" (IEEE
Trans. Inf. Theory 57(10), 6463-6486, 2011, DOI 10.1109/TIT.2011.2165816).

Why it exists
-------------
Tier 1 (docs/hello-world-aircomp.md) demonstrates ANALOG superposition: two
transmitters send pre-coded signals, the receiver reads their real-valued
sum from the superposed waveform. That is AirComp's pedagogical core — but
the result is an ESTIMATE: noise corrupts it irrecoverably, and the only
alternative is routing — decoding every message individually first.

The Nazer/Gastpar result says something stronger: with NESTED LATTICE codes,
the same channel superposition can compute an EXACT function of the messages
(the integer sum, mod L) at finite SNR — with error probability decaying
exponentially in the lattice dimension, at a rate no worse than the
single-user channel. Interference is not merely tolerated; it is the
operation, and the lattice structure makes it exact.

The construction, made legible
-----------------------------
A lattice is a discrete additive subgroup of R^n. The pair used here is the
simplest nested family:

    fine lattice   = Z^n        (the integers — message carriers)
    coarse lattice = L·Z^n      (the fundamental cell [-L/2, L/2)^n)

Each node i holds an integer message w_i in Z_L = {0..L-1}. The message
labels a COSET of the coarse lattice inside the fine lattice: the set
{w_i + L·k : k in Z^n}. Encoding picks a random coset representative
v_i = w_i + L·k_i (random coding, Shannon-style), subtracts a dither d_i
uniform on the coarse cell, and reduces mod L:

    x_i = [v_i - d_i] mod L        (uniform on the cell — "lattice-like")

The dither does two jobs, both from Erez & Zamir (2004), on which Nazer/
Gastpar build: it makes the transmitted signal uniform over the cell (power
shaping) and statistically independent of the message (the transmitted
waveform alone reveals nothing about w_i). The dither is NOT secret — in
the theory it is generated from a seed shared with the receiver, which
subtracts it. What the receiver never learns is the messages, the coset
representatives, or any individual transmission.

The channel superposes all nodes and adds noise:

    y = sum_i h_i * x_i + z

With the channel-inverted gains h_i = 1 (Tier 1 pre-coding), expand the mod
operations: y + sum_i d_i = sum_i v_i + L·m + z (m an integer vector), and
sum_i v_i = sum_i w_i + L·K — the message SUM rides the fine lattice.
Reduce mod L and the coarse components are gone (mod), the coset randomness
is gone (mod), the dithers are gone (replayed — added back by the receiver
from the shared seed, per Erez-Zamir dithered coding), and the noise
survives only as a perturbation of an INTEGER. Rounding an integer plus noise
recovers the integer exactly unless |z| crossed the half-integer boundary —
a hard threshold. Across the n coordinates of the lattice, each carrying an
independent noisy copy of the same sum, the decoder AVERAGES before it
rounds: the full sqrt(n) dimensional gain, error probability decaying
exponentially in n (this is the toy-scale version of the random-lattice
error exponent in the theorem).

What the receiver gets: sum_i w_i mod L — exact with high probability —
from ONE channel use, having decoded NO individual message.

Why nested lattices and not plain repetition
--------------------------------------------
The naive scheme ("just transmit w_i and average the noise") fails at the
wrap: Z_L is a CIRCLE, not an interval, so plain averaging suffers a
circular-wrap failure at larger L (values near 0 and L-1 pull the mean in
opposite directions). The lattice construction moves the wrap into the
mod-L reduction, where it is harmless: each coordinate becomes a point on
the unit circle carrying the SAME angular offset sum·(2π/L), and the
decoder takes the CIRCULAR MEAN — average the unit vectors, read the angle.
That recovers the full averaging gain at any L. This is the pedagogical
heart of the module: the lattice doesn't change the channel; it changes
what the noise is ALLOWED to do.

The honest scoreline (what the Monte Carlo measures)
-----------------------------------------------------
At matched transmit power and matched per-coordinate SNR:

  1. EXACTNESS. Lattice decoding is exact with error probability that decays
     exponentially in n. The Tier 1 analog estimate has irreducible MSE
     sigma^2/n — fine as an estimate, useless as arithmetic.
  2. RESOURCE USE. Routing (TDMA: decode each message, then add) needs N
     channel uses for N nodes and unions error across slots. Lattice AirComp
     needs ONE channel use — the sum is what's decodable, not the parts.
  3. PRIVACY. The receiver recovers only sum w_i mod L. Individual w_i never
     exist at the receiver — not masked, not encoded-around: structurally
     absent. Analog+round at matched power has comparable error rates (the
     single-user rate is the ceiling for both — that IS the theorem), but
     transmits raw values and computes no function class.

Toy scope (marked inline): the theory uses high-dimensional random lattice
ensembles; we use repetition across n coordinates, which already exhibits
the exponential error decay. The theory optimizes the receiver scaling
alpha; we fix alpha = 1 (inverted channel) and expose only integer
coefficients. The 2007 paper computes over FINITE fields; the toy's Z_L is
the integers mod L, the cleanest finite ring for pedagogy.

Module map
----------
    mod_lattice(x, cell)        — reduce into the coarse fundamental cell
    encode(w, L, n, rng)        — message -> dithered lattice codeword
    channel(xs, L, snr_db, ..)  — superpose + AWGN (the wave domain)
    decode(y, L, dithers, a)   — superposed signal -> integer sum mod L
    run_trial(...)              — one AirComp round, all three schemes
    monte_carlo(...)            — measure the scoreline over many trials
"""

from __future__ import annotations

import numpy as np

__all__ = [
    'mod_lattice', 'encode', 'decode', 'channel', 'run_trial', 'monte_carlo',
]


# ─────────────────────────────────────────────────────────────────────────────
# The nested lattice pair: fine = Z^n, coarse = L·Z^n
# ─────────────────────────────────────────────────────────────────────────────

def mod_lattice(x, cell):
    """
    Reduce x into the fundamental cell [-cell/2, cell/2)^n of lattice cell·Z^n.

    The lattice analogue of integer "mod": discards the lattice component,
    keeps the residue. Used by the encoder for power shaping and by the
    decoder to strip everything except message-sum + noise.
    """
    x = np.asarray(x, dtype=np.float64)
    return x - cell * np.round(x / cell)


def encode(w, L, n, rng):
    """
    Encode integer message w in {0..L-1} into a dithered fine-lattice codeword.

    Construction (per node):
      v = w + L·k   with random k in {0..L-1}^n   — random coset representative
      d = uniform on [-L/2, L/2)^n               — the dither (shared w/ receiver)
      x = [v - d] mod L                          — the transmitted signal

    x is uniform on the coarse cell REGARDLESS of w (Erez-Zamir dithered
    lattice coding) — power-shaped and message-independent.

    Returns (x, d). The dither must reach the receiver (in practice: a shared
    PRNG seed; in this simulation: passed alongside). The MESSAGE never
    leaves the node.
    """
    w = int(w)
    L = int(L)
    if not (0 <= w < L):
        raise ValueError("message w=%d out of range [0, %d)" % (w, L))
    v = np.full(n, float(w)) + L * rng.integers(0, L, size=n)   # fine coset of w
    d = rng.uniform(-L / 2, L / 2, size=n)                       # dither
    x = mod_lattice(v - d, float(L))                              # transmit
    return x, d


def channel(xs, L, snr_db, gains=None):
    """
    The wave domain: superpose all nodes with channel gains, add AWGN.

        y = sum_i h_i · x_i + z,    z ~ N(0, sigma^2 I)

    snr_db is per-node. Each x_i is uniform on a cell of side L, so its
    per-coordinate second moment is L^2/12 — that IS the per-node signal
    power, and sigma^2 = (L^2/12) / snr.

    With gains=None (default) the Tier 1 channel-inversion pre-coding is
    assumed: h_i = 1 at the receiver. Arbitrary gains model a real channel;
    decode(a=...) then absorbs them into an integer coefficient vector.
    """
    if not xs:
        raise ValueError("channel needs at least one codeword")
    n = len(xs[0])
    if gains is None:
        gains = [1.0] * len(xs)
    y = np.zeros(n)
    for h, x in zip(gains, xs):
        y += h * x
    P = (float(L) ** 2) / 12.0          # per-node signal power (uniform cell)
    snr = 10 ** (snr_db / 10)
    sigma = np.sqrt(P / snr)
    y = y + np.random.default_rng().normal(0.0, sigma, size=n)
    return y


def decode(y, L, dithers, a=None):
    """
    Decode the superposed signal y into the integer combination sum a_i·w_i, mod L.

    The receiver knows: the codebook geometry (L), the dithers (shared
    seeds), and its own coefficient intent a. It never knows the messages
    w_i, the coset representatives k_i, or any individual x_i.

    The algebra (the heart of the construction). Encoding gives, per node:
        x_i = v_i - d_i - L·m_i        (v_i = w_i + L·k_i, m_i the wraps)
    The channel superposes with gains:  y = Σ h_i·x_i + z. The receiver
    replays the dithers weighted by the effective integer gains a_i
    (with a_i = h_i the replay is exact — that is why integer coefficients
    are free):
        y + Σ a_i·d_i = Σ a_i·w_i + L·(integers) + z
    Every L-multiple dies in the mod reduction, and what survives is
        r = [Σ a_i·w_i + z] mod L
    — the requested integer combination of the messages, plus noise, cleanly.
    The dithers are not "stripped"; they are REPLAYED at the coefficient
    weights (Erez-Zamir dithered coding: dither shapes the transmit signal;
    the replay recovers the combination).

    Then the noise: each of the n coordinates is an independent noisy copy
    of the same integer combination, wrapped mod L. Plain averaging fails
    on the wrap (values near 0 and L-1 pull in opposite directions);
    round-then-vote forfeits the dimensional gain. The circular mean keeps
    both: map each r_j to its unit vector at angle 2π·r_j/L, average the
    vectors, read the angle. Full sqrt(n) gain at any L.

    Steps:
      1. replay dithers:    y' = y + Σ a_i·d_i
      2. mod-reduce:        r = y' mod L
      3. circular average:  unit vectors at 2π·r/L, mean, angle
      4. round + mod:       s_hat = round(angle·L/2π) mod L

    a must be integer-valued (the gains the receiver replays against).
    Non-integer a needs the theory's alpha optimization and is out of toy
    scope. With a = all-ones this is the plain sum (Nazer/Gastpar 2007);
    with integer a it is the compute-and-forward equation (2011).
    """
    y = np.asarray(y, dtype=np.float64)
    num_nodes = len(dithers)
    if a is None:
        a = np.ones(num_nodes)
    a = np.asarray(a, dtype=np.float64)
    # 1. replay the dithers at the coefficient weights
    y_prime = y.copy()
    for i in range(num_nodes):
        y_prime = y_prime + a[i] * dithers[i]
    # 2. mod-lattice reduction kills every coarse component
    r = mod_lattice(y_prime, float(L))
    # 3. circular mean across the n coordinates
    theta = 2 * np.pi * r / float(L)             # each coordinate: same integer, mod-L wrapped
    m = np.exp(1j * theta)                        # unit vectors on the mod-L circle
    m_bar = m.mean()                              # the averaging — noise cancels here
    angle = np.angle(m_bar)                       # in (-pi, pi]
    s_hat = np.rint(angle * float(L) / (2 * np.pi)) % L
    return int(s_hat)


# ─────────────────────────────────────────────────────────────────────────────
# One AirComp round, three schemes side by side
# ─────────────────────────────────────────────────────────────────────────────

def run_trial(num_nodes=2, L=16, n=32, snr_db=5.0, a=None, rng=None, gains=None):
    """
    Run ONE AirComp round with the same channel statistics for all schemes:

      lattice — nested-lattice coded (this module): exact sum mod L, 1 channel use
      analog  — Tier 1 analog superposition + rounding: 1 channel use, estimate
      tdma    — routing baseline: N sequential slots, exact per-slot decode, N uses

    Returns per-scheme decoded value, the true value, and exact-match booleans.
    """
    if rng is None:
        rng = np.random.default_rng()
    if a is None:
        a = np.ones(num_nodes)
    ws = rng.integers(0, L, size=num_nodes)
    true_val = int(np.sum(np.asarray(a) * ws) % L)

    # ── lattice scheme ──
    xs, ds = [], []
    for w in ws:
        x, d = encode(int(w), L, n, rng)
        xs.append(x)
        ds.append(d)
    y = channel(xs, L, snr_db, gains=gains)
    lattice_val = decode(y, L, ds, a=a)

    # ── analog scheme (Tier 1 baseline) ──
    # Raw-value transmission, scaled to MATCHED average power. The lattice's
    # dithered codeword is uniform on the cell: power L^2/12 with the full
    # cell as noise margin. Raw w_i on {0..L-1} has second moment ~L^2/3 —
    # 4x the power for the same 1-message-unit resolution. Matching power
    # (scale = 1/2) halves analog's noise margin: the constellation is not
    # shaped to the cell. This power-shaping gap — plus analog's irreducible
    # MSE and its inability to represent mod-L arithmetic — is the honest
    # difference. (The data-dependent power of a raw-value sum is exactly
    # the "power alignment" problem the field map defers.)
    P = (L ** 2) / 12.0
    snr = 10 ** (snr_db / 10)
    sigma = np.sqrt(P / snr)
    analog_scale = 0.5                      # matched average power: sqrt((L^2/12)/(L^2/3))
    # n independent copies of the sum, noise averaged — the analog scheme's
    # best shot at the same dimensional gain
    per_coord = float(np.sum(ws)) * analog_scale + rng.normal(0.0, sigma, size=n)
    est = per_coord.mean() / analog_scale
    analog_val = int(np.rint(est)) % L

    # ── tdma scheme ──
    # Routing baseline at FULL power shaping: each slot centers its value in
    # the cell ((w - L/2) has second moment ~L^2/12 — the same shaping the
    # lattice gets for free from its dither), n repeats averaged per slot,
    # each message decoded exactly, then summed. This is TDMA's best case:
    # per-slot reliability ~ the lattice's, at the cost of N channel uses
    # and an N-way error union. The near-parity at N times the resources IS
    # the Nazer/Gastpar computation-rate result, made visible.
    tdma_val = 0
    for w in ws:
        slot_tx = (float(w) - L / 2) + rng.normal(0.0, sigma, size=n)
        est_slot = slot_tx.mean() + L / 2
        tdma_val += int(np.rint(est_slot))
    tdma_val = int(tdma_val) % L

    return {
        'messages': [int(w) for w in ws],
        'true_value': true_val,
        'lattice': {'value': lattice_val, 'exact': lattice_val == true_val},
        'analog': {'value': analog_val, 'exact': analog_val == true_val},
        'tdma': {'value': tdma_val, 'exact': tdma_val == true_val},
        'channel_uses': {'lattice': 1, 'analog': 1, 'tdma': int(num_nodes)},
    }


def monte_carlo(num_nodes=2, L=16, n=32, snr_db=5.0, trials=2000, seed=42):
    """
    The Tier 1.5 experiment: measure all three schemes over many trials.

    The honest scoreline, as measured:
      - LATTICE: 1 channel use. Exact sum mod L with error probability p
        that decays exponentially in the dimension n.
      - TDMA (routing): N channel uses — and its error is the UNION of N
        per-slot errors, each statistically like the lattice's: observed
        error ≈ N·p. It pays N times the resources to do WORSE than the
        lattice's single use. This is the Nazer/Gastpar computation-rate
        result made visible: the sum is decodable at the rate of ONE user
        while all N transmit.
      - ANALOG (Tier 1): 1 channel use, but strictly worse error at every
        SNR — raw values are not power-shaped to the cell (a factor-2 noise
        margin gap at matched power), and a real-valued estimate cannot
        represent mod-L arithmetic at all.
      - PRIVACY (not in the numbers): the lattice receiver recovers ONLY
        sum w_i mod L — individual messages are structurally absent, not
        masked. Analog transmits raw values; TDMA decodes every message.
    """
    rng = np.random.default_rng(seed)
    lat_err = ana_err = tdma_err = 0
    for _ in range(trials):
        r = run_trial(num_nodes=num_nodes, L=L, n=n, snr_db=snr_db, rng=rng)
        lat_err += not r['lattice']['exact']
        ana_err += not r['analog']['exact']
        tdma_err += not r['tdma']['exact']
    return {
        'trials': trials,
        'snr_db': snr_db,
        'lattice': {'error_rate': lat_err / trials, 'channel_uses_per_trial': 1},
        'analog': {'error_rate': ana_err / trials, 'channel_uses_per_trial': 1},
        'tdma': {'error_rate': tdma_err / trials, 'channel_uses_per_trial': num_nodes},
    }