"""
Coefficient selection for compute-and-forward — the fading-channel tier
========================================================================

Tier 1.5 (rf_compute/lattice.py) SETS the channel gains to integers
(h_i = a_i), which makes the compute-and-forward equation exact — but it
hides the problem the coefficient vector exists to solve: on a real channel
the gains are FADING (real-valued, not integers), and the receiver must
CHOOSE an integer vector a whose combination sum(a_i * w_i) mod L it wants
to decode.

This module implements the selection machinery:

    computation_rate(h, a, snr_db)      — the Nazer/Gastpar rate for one a
    mmse_alpha(h, a, snr_db)            — the optimal receiver scaling alpha
    select_coefficients(h, snr_db, ..)  — exhaustive (norm bound) / LLL / rounded
    fading_gains(num_nodes, rng)        — a real fading realization
    lll_reduce(basis)                   — Lenstra-Lenstra-Lovasz reduction

The theory in one paragraph
---------------------------
A real channel y = sum_i h_i x_i + z. The receiver scales by alpha and
replays the shared dithers weighted at the effective gains alpha*h_i:

    y' = alpha*y + sum_i alpha*h_i*d_i
       = sum_i alpha*h_i*v_i + L*(integers) + alpha*z

Mod-L reduction kills the lattice components; what remains is the real
number sum(alpha*h_i*w_i) + alpha*z, which the decoder rounds to the
nearest integer — the combination sum(a_i*w_i) mod L, PROVIDED alpha*h_i
is close to the integers a_i. The mismatch is the self-noise:

    Z_eff(alpha, a) = alpha^2 sigma^2 + P * sum_i (alpha*h_i - a_i)^2

with P the per-node transmit power (L^2/12, the uniform-cell convention)
and sigma^2 = P / SNR. The computation rate is

    R(alpha, a) = 1/2 log2^+ ( P / Z_eff )

maximized over alpha and a. Two facts make this tractable:

1. The optimal scaling is the MMSE choice (closed form):
       alpha* = SNR * (h^T a) / (1 + SNR * ||h||^2)
2. Substituting alpha* gives the shortest-lattice-vector (SLV) problem
       minimize D(a) = a^T (I + SNR * h h^T)^{-1} a   over integer a != 0

Sanity check that pins the normalization: N = 1, h = 1, a = 1 gives
D = 1/(1+SNR) and R = 1/2 log2(1 + SNR) — the AWGN capacity of a real
channel, exactly. (At high SNR the integer alignment alpha*h ~ a can only
be perfect when a is parallel to h — the residual sum_i (alpha*h_i - a_i)^2
is the self-noise floor. This is why the real-channel toy never reaches the
cooperative bound for N >= 2 unless h is integer-aligned, and why the
literature works over COMPLEX channels (two real dimensions; Gaussian-
integer lattices align far better) — Liu & Ling 2016.)

Search methods (the literature's lineage, in order)
---------------------------------------------------
- exhaustive within the norm bound ||a|| <= sqrt(1 + SNR ||h||^2):
  Nazer & Gastpar 2011 (the theorem's own bound; exact, exponential in N)
- LLL lattice reduction: approximate the SLV instance in polynomial time.
  The lineage: Sahraei & Gastpar 2014 (exact polynomial algorithm),
  Liu & Ling 2016 (complex channels, efficient integer search)
- rounded: a = round(alpha*h) — the naive nearest-integer heuristic,
  kept as the baseline (it is what you get if you never optimize)

No hardware, no secrets — pure NumPy.
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = [
    'mmse_alpha', 'computation_rate', 'norm_bound', 'select_coefficients',
    'fading_gains', 'lll_reduce',
]


# ─────────────────────────────────────────────────────────────────────────────
# The rate and the scaling
# ─────────────────────────────────────────────────────────────────────────────

def mmse_alpha(h, a, snr_db):
    """
    The MMSE receiver scaling: alpha* = SNR * (h^T a) / (1 + SNR ||h||^2).

    Minimizes the effective decoder noise Z_eff(alpha) = alpha^2 sigma^2 +
    P * sum_i (alpha*h_i - a_i)^2 (Nazer & Gastpar 2011). Returns 0.0 when
    h^T a <= 0 (no positive scaling helps — the rate is 0).
    """
    h = np.asarray(h, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    snr = 10 ** (snr_db / 10)
    num = snr * float(h @ a)
    if num <= 0:
        return 0.0
    return num / (1.0 + snr * float(h @ h))


def computation_rate(h, a, snr_db, alpha=None):
    """
    The compute-and-forward computation rate for coefficient vector a:
        R = 1/2 log2^+ ( P / Z_eff ),    Z_eff = alpha^2 sigma^2 + P ||alpha h - a||^2
    reported in bits per real channel use, with alpha = MMSE unless given.

    Normalization: per-node transmit power P, sigma^2 = P / SNR — the
    per-user power convention of the AirComp literature (Huang & Burr 2017).
    N = 1, h = 1, a = 1 returns exactly 1/2 log2(1 + SNR).
    """
    h = np.asarray(h, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    snr = 10 ** (snr_db / 10)
    if alpha is None:
        alpha = mmse_alpha(h, a, snr_db)
    if alpha <= 0:
        return 0.0
    # Z_eff / P = alpha^2 / SNR + ||alpha h - a||^2
    d = alpha ** 2 / snr + float(np.sum((alpha * h - a) ** 2))
    if d <= 0:
        return 0.0
    return 0.5 * max(0.0, float(np.log2(1.0 / d)))


def norm_bound(h, snr_db):
    """
    The Nazer/Gastpar search bound: any rate-maximizing a satisfies
    ||a|| <= sqrt(1 + SNR ||h||^2).
    """
    h = np.asarray(h, dtype=np.float64)
    snr = 10 ** (snr_db / 10)
    return float(np.sqrt(1.0 + snr * float(h @ h)))


# ─────────────────────────────────────────────────────────────────────────────
# The fading realization
# ─────────────────────────────────────────────────────────────────────────────

def fading_gains(num_nodes, rng, scale=1.0):
    """
    One real fading realization: h_i ~ N(0, scale^2), i.i.d.

    Real baseband fading — the complex case (Gaussian-integer lattices)
    is the literature's practical choice (Liu & Ling 2016) and the natural
    extension here.
    """
    return rng.normal(0.0, scale, size=int(num_nodes))


# ─────────────────────────────────────────────────────────────────────────────
# Selection — the three methods
# ─────────────────────────────────────────────────────────────────────────────

def _rounded_candidates(h, snr_db, iterations=6):
    """
    Nearest-integer heuristic: iterate a <- round(alpha(a) * h).

    Seeded from sign(h) (so h^T a = ||h||_1 > 0 and alpha > 0) and from each
    signed coordinate axis; keeps the best rate seen. Seeds matter: the
    all-ones seed degenerates on a mixed-sign channel (h^T a <= 0 -> alpha
    = 0), which is exactly the case selection exists to handle.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    best = None
    best_rate = 0.0

    def check(a):
        nonlocal best, best_rate
        a = np.asarray(a, dtype=int)
        if not np.any(a):
            return
        rate = computation_rate(h, a, snr_db)
        if best is None or rate > best_rate:
            best, best_rate = a, rate

    seeds = [np.sign(h).astype(int)]
    for i in range(n):
        e = np.zeros(n, dtype=int)
        e[i] = int(np.sign(h[i]) or 1)
        seeds.append(e)

    for seed in seeds:
        a = seed
        check(a)
        for _ in range(iterations):
            alpha = mmse_alpha(h, a, snr_db)
            if alpha <= 0:
                break
            a_new = np.rint(alpha * h).astype(int)
            if not np.any(a_new) or np.array_equal(a_new, a):
                break
            a = a_new
            check(a)

    if best is None:
        best = np.zeros(n, dtype=int)
        i = int(np.argmax(np.abs(h)))
        best[i] = int(np.sign(h[i]) or 1)
        best_rate = computation_rate(h, best, snr_db)
    return best, best_rate


def _exhaustive(h, snr_db, max_component=5, max_vectors=300_000):
    """Exhaustive search of the integer norm ball ||a|| <= bound."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    bound = norm_bound(h, snr_db)
    radius = min(int(np.floor(bound)), int(max_component))
    while radius >= 1 and (2 * radius + 1) ** n > max_vectors:
        radius -= 1
    best = None
    best_rate = 0.0
    for a in itertools.product(range(-radius, radius + 1), repeat=n):
        a_arr = np.asarray(a, dtype=float)
        if not np.any(a_arr):
            continue
        if float(np.linalg.norm(a_arr)) > bound + 1e-9:
            continue
        rate = computation_rate(h, a_arr, snr_db)
        if best is None or rate > best_rate:
            best, best_rate = np.asarray(a, dtype=int), rate
    if best is None:
        best, best_rate = _rounded_candidates(h, snr_db)
    return best, best_rate


def lll_reduce(basis, delta=0.75):
    """
    Classic LLL (Lenstra-Lenstra-Lovasz) basis reduction on a real basis.

    basis: (n x n) array whose ROWS span the lattice. Returns
    (reduced_basis, U) with reduced_basis = U @ basis and U unimodular
    integer (det = +/-1). Textbook algorithm with Gram-Schmidt recomputed
    per step — O(n^3) per op, trivial at toy dimensions.
    """
    B = np.array(basis, dtype=np.float64)
    n = len(B)
    U = np.eye(n, dtype=int)

    def gram_schmidt():
        Bs = np.zeros_like(B)
        mu = np.zeros((n, n))
        for i in range(n):
            Bs[i] = B[i]
            for j in range(i):
                mu[i, j] = (B[i] @ Bs[j]) / (Bs[j] @ Bs[j])
                Bs[i] = Bs[i] - mu[i, j] * Bs[j]
        return Bs, mu

    Bs, mu = gram_schmidt()
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            q = int(round(mu[k, j]))
            if q:
                B[k] = B[k] - q * B[j]
                U[k] = U[k] - q * U[j]
                Bs, mu = gram_schmidt()
        if (Bs[k] @ Bs[k]) >= (delta - mu[k, k - 1] ** 2) * (Bs[k - 1] @ Bs[k - 1]):
            k += 1
        else:
            B[[k - 1, k]] = B[[k, k - 1]]
            U[[k - 1, k]] = U[[k, k - 1]]
            Bs, mu = gram_schmidt()
            k = max(k - 1, 1)
    return B, U


def _lll(h, snr_db):
    """
    LLL-aided selection: reduce the SLV lattice, then enumerate integer
    coordinates in {-1,0,1} of the REDUCED basis (a bounded, small search —
    the LLL-aided enumeration pattern of Liu & Ling 2016; the polynomial-
    complexity exact algorithm is Sahraei & Gastpar 2014).

    D(a) = a^T M a with M = (I + SNR h h^T)^{-1} = I - c h h^T,
    c = SNR/(1 + SNR ||h||^2). Factor M = L L^T (Cholesky); then
    D(a) = ||L^T a||^2 — the squared length of the lattice point
    sum_j a_j * row_j(L). Reduce the row basis; reduced rows U_k are
    integer combinations to enumerate over: a = sum_k c_k U_k.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    snr = 10 ** (snr_db / 10)
    c = snr / (1.0 + snr * float(h @ h))
    M = np.eye(n) - c * np.outer(h, h)
    L = np.linalg.cholesky(M)
    _, U = lll_reduce(L)
    best = None
    best_rate = 0.0
    for coeffs in itertools.product([-1, 0, 1], repeat=n):
        a = np.asarray(coeffs, dtype=int) @ U
        if not np.any(a):
            continue
        rate = computation_rate(h, a, snr_db)
        if best is None or rate > best_rate:
            best, best_rate = np.array(a, dtype=int), rate
    if best is None:
        best, best_rate = _rounded_candidates(h, snr_db)
    return best, best_rate


def select_coefficients(h, snr_db, method="exhaustive"):
    """
    Choose the integer coefficient vector a maximizing the computation rate.

    h: real channel gains. snr_db: per-node SNR (P / sigma^2, P = L^2/12).

    method:
      'exhaustive' — search the Nazer/Gastpar norm ball (exact, exponential)
      'lll'        — LLL-reduced candidate rows (polynomial-time approximation)
      'rounded'    — nearest-integer heuristic round(alpha * h) (baseline)

    Returns (a, rate) with a an integer numpy array, never all-zero.
    """
    h = np.asarray(h, dtype=np.float64)
    if method == "exhaustive":
        return _exhaustive(h, snr_db)
    if method == "lll":
        return _lll(h, snr_db)
    if method == "rounded":
        return _rounded_candidates(h, snr_db)
    raise ValueError(
        f"unknown selection method '{method}'; "
        "use 'exhaustive', 'lll', or 'rounded'"
    )
