"""Tests for the compute-and-forward coefficient selection machinery.

Covers rf_compute.coefficients: the MMSE scaling, the computation rate
(with its exact single-user pin), the Nazer/Gastpar norm bound, the three
selection methods, fading realizations, and LLL reduction.
"""

import numpy as np
import pytest

from rf_compute.coefficients import (
    mmse_alpha, computation_rate, norm_bound, select_coefficients,
    fading_gains, lll_reduce,
)


# ─── the rate formula's normalization pins ──────────────────────────────

def test_single_user_rate_equals_awgn_capacity():
    # N=1, h=1, a=1, MMSE alpha: R = 1/2 log2(1 + SNR), EXACTLY.
    # This is the pin that fixes the whole normalization chain
    # (D(a) via Sherman-Morrison, E_min = P D(a), R = 1/2 log+ (P/E)).
    for snr_db in [0.0, 5.0, 10.0, 20.0]:
        snr = 10 ** (snr_db / 10)
        r = computation_rate(np.array([1.0]), np.array([1]), snr_db)
        assert abs(r - 0.5 * np.log2(1.0 + snr)) < 1e-12


def test_mmse_alpha_improves_the_rate():
    # On a unit-gain channel the MMSE scaling sits below 1 and strictly
    # beats alpha = 1 — the reason the theory optimizes alpha at all.
    h, a = np.array([1.0]), np.array([1])
    alpha = mmse_alpha(h, a, 5.0)
    assert 0.0 < alpha < 1.0
    assert computation_rate(h, a, 5.0, alpha=alpha) > computation_rate(h, a, 5.0, alpha=1.0)


def test_mmse_alpha_zero_when_misaligned():
    # h^T a <= 0: no positive scaling helps; the rate is zero.
    h, a = np.array([1.0, -1.0]), np.array([1, 1])
    assert mmse_alpha(h, a, 5.0) == 0.0
    assert computation_rate(h, a, 5.0) == 0.0


def test_computation_rate_is_nonnegative():
    rng = np.random.default_rng(0)
    for _ in range(30):
        h = fading_gains(4, rng)
        a = rng.integers(-2, 3, size=4)
        assert computation_rate(h, a, 3.0) >= 0.0


def test_norm_bound_matches_theorem_formula():
    # ||a|| <= sqrt(1 + SNR ||h||^2) — Nazer & Gastpar 2011.
    h = np.array([0.7, -1.3])
    snr_db = 10.0
    snr = 10 ** (snr_db / 10)
    expected = np.sqrt(1.0 + snr * float(h @ h))
    assert abs(norm_bound(h, snr_db) - expected) < 1e-12


# ─── selection methods ──────────────────────────────────────────────────

def test_exhaustive_beats_or_ties_rounded():
    # Exhaustive searches the norm ball; the nearest-integer heuristic can
    # miss the optimum, so exhaustive must never do worse.
    rng = np.random.default_rng(3)
    for _ in range(25):
        h = fading_gains(3, rng)
        _, r_ex = select_coefficients(h, 5.0, method="exhaustive")
        _, r_rd = select_coefficients(h, 5.0, method="rounded")
        assert r_ex >= r_rd - 1e-12


def test_exhaustive_beats_or_ties_lll():
    rng = np.random.default_rng(4)
    for _ in range(25):
        h = fading_gains(3, rng)
        _, r_ex = select_coefficients(h, 5.0, method="exhaustive")
        _, r_ll = select_coefficients(h, 5.0, method="lll")
        assert r_ex >= r_ll - 1e-12


def test_exhaustive_satisfies_the_norm_bound():
    # The searched vector must sit inside the theorem's bound (the bound is
    # what makes the exhaustive search finite).
    rng = np.random.default_rng(5)
    for _ in range(20):
        h = fading_gains(3, rng)
        a, _ = select_coefficients(h, 5.0, method="exhaustive")
        assert np.linalg.norm(a) <= norm_bound(h, 5.0) + 1e-9


def test_selection_never_returns_all_zero():
    rng = np.random.default_rng(6)
    for method in ["exhaustive", "lll", "rounded"]:
        for _ in range(20):
            h = fading_gains(3, rng)
            a, rate = select_coefficients(h, 5.0, method=method)
            assert np.any(a), method
            assert rate >= 0.0


def test_rounded_optimal_on_an_aligned_channel():
    # A positive integer-aligned channel is the heuristic's home turf: it
    # must reach the same rate as the exhaustive search. NOTE the optimum is
    # a = [0, 1] (drop the weaker node), not [1, 2]: since MMSE alpha < 1,
    # even an integer-aligned channel cannot align perfectly — the weaker
    # node's self-noise outweighs its contribution. Selection finds that;
    # this is why coefficient selection is a real optimization, not
    # round(alpha*h) by another name.
    h = np.array([1.0, 2.0])
    a_rd, r_rd = select_coefficients(h, 5.0, method="rounded")
    _, r_ex = select_coefficients(h, 5.0, method="exhaustive")
    assert r_rd > 0.0
    assert abs(r_rd - r_ex) < 1e-9
    assert np.any(a_rd)


def test_integer_aligned_channel_still_faces_a_self_noise_floor():
    # The module's honest statement, pinned as a test: on h = [1, 2] the
    # full combination a = [1, 2] scores BELOW the single strongest node —
    # a real channel never reaches the cooperative (perfect-alignment)
    # bound for N >= 2 unless h is parallel to a and alpha can be 1.
    h = np.array([1.0, 2.0])
    r_full = computation_rate(h, np.array([1, 2]), 5.0)
    r_single = computation_rate(h, np.array([0, 1]), 5.0)
    assert r_single > r_full


def test_selection_method_rejects_unknown():
    with pytest.raises(ValueError):
        select_coefficients(np.array([1.0]), 5.0, method="magic")


def test_selection_is_deterministic():
    h = np.array([0.4, -1.1, 0.9])
    for method in ["exhaustive", "lll", "rounded"]:
        a1, r1 = select_coefficients(h, 5.0, method=method)
        a2, r2 = select_coefficients(h, 5.0, method=method)
        assert list(a1) == list(a2) and r1 == r2


# ─── fading ─────────────────────────────────────────────────────────────

def test_fading_gains_shape_and_scale():
    rng = np.random.default_rng(7)
    h = fading_gains(5, rng)
    assert h.shape == (5,)
    assert h.dtype == np.float64
    big = fading_gains(20000, rng, scale=2.0)
    assert 1.7 < np.std(big) < 2.3


def test_fading_gains_reproducible():
    a = fading_gains(4, np.random.default_rng(9))
    b = fading_gains(4, np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)


# ─── LLL ────────────────────────────────────────────────────────────────

def test_lll_is_unimodular_and_consistent():
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(30):
        B = rng.integers(-5, 6, size=(4, 4)).astype(float)
        if abs(np.linalg.det(B)) < 1e-9:
            continue
        Rb, U = lll_reduce(B)
        assert abs(abs(np.linalg.det(U)) - 1) < 1e-9      # unimodular
        np.testing.assert_allclose(Rb, U @ B)              # same lattice
        checked += 1
    assert checked >= 20


def test_lll_shortens_the_shortest_vector():
    rng = np.random.default_rng(13)
    checked = 0
    for _ in range(20):
        B = rng.integers(-9, 10, size=(5, 5)).astype(float)
        if abs(np.linalg.det(B)) < 1e-9:
            continue
        Rb, _ = lll_reduce(B)
        # The reduced first row is no longer than the original first row,
        # and typically much shorter (that is the point of reduction).
        assert np.linalg.norm(Rb[0]) <= np.linalg.norm(B[0]) + 1e-9
        checked += 1
    assert checked >= 15


def test_lll_identity_like_basis_unchanged():
    B = np.eye(3)
    Rb, U = lll_reduce(B)
    np.testing.assert_allclose(Rb, np.eye(3))
    np.testing.assert_allclose(U, np.eye(3, dtype=int))
