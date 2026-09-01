"""Tests for the Tier 1.5 nested-lattice AirComp machinery.

Covers rf_compute.lattice (encode/channel/decode, run_trial, monte_carlo)
and the LatticeAirCompOperator kernel integration.
"""

import numpy as np
import pytest

from rf_compute.lattice import (
    mod_lattice, encode, decode, channel, run_trial, monte_carlo,
)
from rf_compute import LatticeAirCompOperator, WaveComputeKernel


# ─── mod_lattice ────────────────────────────────────────────────────────

def test_mod_lattice_maps_into_fundamental_cell():
    for x in [-17.0, -8.3, -4.0, -0.1, 0.0, 0.1, 3.9, 7.5, 16.2, 33.0]:
        r = mod_lattice(np.array([x]), 8.0)
        assert -4.0 <= r[0] < 4.0


def test_mod_lattice_is_idempotent():
    x = np.array([3.3, -6.7, 12.1])
    once = mod_lattice(x, 5.0)
    twice = mod_lattice(once, 5.0)
    np.testing.assert_allclose(once, twice)


def test_mod_lattice_coset_invariance():
    # x and x + k*cell reduce to the same point — the lattice's translation
    # invariance that makes the coset (not the point) the message.
    x = np.array([1.7, -2.3, 4.4])
    shifted = mod_lattice(x + 3 * 9.0, 9.0)
    np.testing.assert_allclose(shifted, mod_lattice(x, 9.0))


# ─── encode ─────────────────────────────────────────────────────────────

def test_encode_shapes_and_power():
    rng = np.random.default_rng(0)
    L, n = 12.0, 64
    for w in [0, 5, 11]:
        x, d = encode(w, L, n, rng)
        assert x.shape == (n,) and d.shape == (n,)
        # Per-coordinate power ~ L^2/12 (uniform-cell second moment)
        assert abs(np.mean(x ** 2) - L ** 2 / 12.0) < 0.35 * L ** 2 / 12.0


def test_encode_dither_is_shared_not_secret():
    # The dither is returned to the caller — it is shared with the receiver,
    # not a secret. The receiver needs it to replay the construction.
    rng = np.random.default_rng(1)
    x, d = encode(4, 16.0, 32, rng)
    assert d is not None and d.shape == (32,)


def test_encode_rejects_out_of_range_message():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        encode(16, 16, 8, rng)


# ─── decode: noise-free exactness ───────────────────────────────────────

def test_decode_exact_without_noise():
    # The heart of the construction: the receiver replays the dithers it was
    # told, and the mod-L reduction strips the integer ambiguity.
    rng = np.random.default_rng(42)
    ok = 0
    for _ in range(200):
        L = 16.0
        ws = rng.integers(0, 16, size=3)
        xs, ds = [], []
        for w in ws:
            x, d = encode(int(w), L, 32, rng)
            xs.append(x)
            ds.append(d)
        y = np.sum(xs, axis=0)
        if decode(y, L, ds) == int(np.sum(ws) % 16):
            ok += 1
    assert ok == 200


def test_decode_uses_dither_replay_not_message_knowledge():
    # decode's signature has no access to the messages: with the dithers
    # alone (plus the channel output) it recovers the sum.
    rng = np.random.default_rng(3)
    L = 16.0
    xs, ds = [], []
    ws = [2, 7, 9]  # sum = 18 -> 2 mod 16
    for w in ws:
        x, d = encode(w, L, 48, rng)
        xs.append(x)
        ds.append(d)
    y = np.sum(xs, axis=0)
    assert decode(y, L, ds) == 2


def test_decode_error_decays_with_dimension():
    # The theorem's signature: error probability decays with lattice
    # dimension n (each coordinate is an independent vote on the sum).
    mc4 = monte_carlo(num_nodes=2, L=16, n=4, snr_db=0.0, trials=200, seed=1)
    mc48 = monte_carlo(num_nodes=2, L=16, n=48, snr_db=0.0, trials=200, seed=1)
    assert mc48['lattice']['error_rate'] < mc4['lattice']['error_rate']


def test_decode_error_decays_with_snr():
    mc_low = monte_carlo(num_nodes=2, L=16, n=32, snr_db=-5.0, trials=300, seed=2)
    mc_high = monte_carlo(num_nodes=2, L=16, n=32, snr_db=10.0, trials=300, seed=2)
    assert mc_high['lattice']['error_rate'] < mc_low['lattice']['error_rate']


def test_circular_mean_keeps_dimensional_gain():
    # The decoder's reason to exist: plain averaging wraps (Z_L is a circle),
    # round-then-vote forfeits the sqrt(n) gain; the circular mean keeps it.
    # Pinned as a property: at 5 dB with n=64 the lattice scheme is reliable.
    mc = monte_carlo(num_nodes=2, L=16, n=64, snr_db=5.0, trials=300, seed=3)
    assert mc['lattice']['error_rate'] < 0.25


# ─── coefficients (compute-and-forward equation) ─────────────────────────

def test_coefficients_as_channel_gains_noise_free():
    # Integer coefficients are realized as channel gains: node i arrives at
    # gain a_i, receiver replays dithers at the same weights.
    rng = np.random.default_rng(5)
    ok = 0
    for _ in range(100):
        L = 16.0
        ws = rng.integers(0, 16, size=3)
        a = [int(c) for c in rng.integers(1, 4, size=3)]
        xs, ds = [], []
        for w, ai in zip(ws, a):
            x, d = encode(int(w), L, 32, rng)
            xs.append(ai * x)   # the gain applied at the channel
            ds.append(d)
        y = np.sum(xs, axis=0)
        truth = int(np.sum(np.asarray(a) * ws) % 16)
        if decode(y, L, ds, a=a) == truth:
            ok += 1
    assert ok == 100


def test_lattice_channel_with_gains_matches_coefficients():
    # lattice.channel(gains=...) applies the same gains the decoder expects.
    rng = np.random.default_rng(6)
    L = 16.0
    ws = [5, 9]
    a = [1, 2]
    xs, ds = [], []
    for w in ws:
        x, d = encode(w, L, 64, rng)
        xs.append(x)
        ds.append(d)
    y = channel(xs, L, snr_db=60.0, gains=a)  # effectively noise-free
    truth = int(np.sum(np.asarray(a) * ws) % 16)
    assert decode(y, L, ds, a=a) == truth


# ─── channel ────────────────────────────────────────────────────────────

def test_channel_adds_noise_at_the_configured_snr():
    rng = np.random.default_rng(8)
    L = 16.0
    xs = [encode(3, L, 512, rng)[0] for _ in range(2)]
    clean = np.sum(xs, axis=0)
    noisy = channel(xs, L, snr_db=0.0)
    resid = noisy - clean
    sigma_expected = np.sqrt((L ** 2 / 12.0) / (10 ** (0.0 / 10.0)))
    sigma_observed = np.std(resid)
    assert 0.6 * sigma_expected < sigma_observed < 1.6 * sigma_expected


def test_channel_rejects_empty_codeword_list():
    with pytest.raises(ValueError):
        channel([], 16.0, snr_db=10.0)


# ─── run_trial / monte_carlo: the honest scoreline ───────────────────────

def test_run_trial_returns_all_three_schemes():
    r = run_trial(num_nodes=2, L=16, n=32, snr_db=10.0)
    for scheme in ['lattice', 'analog', 'tdma']:
        assert scheme in r
        assert 'value' in r[scheme] and 'exact' in r[scheme]
    # The resource story: lattice and analog are one channel use, TDMA pays N.
    assert r['channel_uses']['lattice'] == 1
    assert r['channel_uses']['analog'] == 1
    assert r['channel_uses']['tdma'] == 2  # == num_nodes


def test_monte_carlo_lattice_beats_both_baselines_at_moderate_snr():
    # The theorem made visible: at matched power, one lattice channel use
    # beats TDMA (which pays N channel uses AND an N-way error union) and
    # analog superposition (which carries no mod-L arithmetic, pays a
    # shaping penalty, and keeps an irreducible MSE).
    mc = monte_carlo(num_nodes=2, L=16, n=32, snr_db=10.0, trials=400, seed=4)
    assert mc['lattice']['error_rate'] < mc['tdma']['error_rate']
    assert mc['lattice']['error_rate'] < mc['analog']['error_rate']


def test_tdma_pays_the_error_union():
    # TDMA decodes each message separately and sums: one bad slot ruins the
    # sum, so its error rate grows with N even as it pays N channel uses.
    mc2 = monte_carlo(num_nodes=2, L=16, n=32, snr_db=5.0, trials=300, seed=5)
    mc6 = monte_carlo(num_nodes=6, L=16, n=32, snr_db=5.0, trials=300, seed=5)
    assert mc6['tdma']['error_rate'] > mc2['tdma']['error_rate']


def test_monte_carlo_channel_uses_accounting():
    mc = monte_carlo(num_nodes=3, L=16, n=32, snr_db=5.0, trials=50, seed=6)
    assert mc['lattice']['channel_uses_per_trial'] == 1
    assert mc['analog']['channel_uses_per_trial'] == 1
    assert mc['tdma']['channel_uses_per_trial'] == 3


# ─── kernel integration ─────────────────────────────────────────────────

def _kernel():
    return WaveComputeKernel()


def test_operator_validates_modulus():
    with pytest.raises(ValueError):
        LatticeAirCompOperator(modulus=1)


def test_operator_validates_dimension():
    with pytest.raises(ValueError):
        LatticeAirCompOperator(dimension=0)


def test_operator_validates_coefficients_length():
    with pytest.raises(ValueError):
        LatticeAirCompOperator(num_nodes=3, coefficients=[1, 2])


def test_operator_accepts_and_ints_coefficients():
    op = LatticeAirCompOperator(num_nodes=2, coefficients=[1.0, 2.0])
    assert op.coefficients == [1, 2]


def test_kernel_plain_sum_exact():
    k = _kernel()
    op = LatticeAirCompOperator(num_nodes=3, modulus=16, dimension=32)
    out = k.apply(op, [3, 7, 11])
    assert out['value'] == (3 + 7 + 11) % 16
    assert out['exact'] is True
    assert out['true_value'] == out['value']


def test_kernel_coefficient_equation_exact():
    # sum(a_i * w_i) mod L — the compute-and-forward equation.
    k = _kernel()
    op = LatticeAirCompOperator(num_nodes=2, modulus=16, dimension=32,
                                coefficients=[1, 2])
    out = k.apply(op, [5, 9])
    assert out['value'] == (1 * 5 + 2 * 9) % 16
    assert out['exact'] is True


def test_kernel_coefficient_sweep_noise_free():
    k = _kernel()
    rng = np.random.default_rng(7)
    ok = 0
    for _ in range(100):
        N, L = 4, 16
        ws = list(rng.integers(0, L, size=N))
        a = list(rng.integers(1, 4, size=N))
        op = LatticeAirCompOperator(num_nodes=N, modulus=L, dimension=32,
                                    coefficients=a)
        r = k.apply(op, ws)
        ok += r['exact']
    assert ok == 100


def test_kernel_rejects_out_of_range_message():
    k = _kernel()
    op = LatticeAirCompOperator(num_nodes=2, modulus=16, dimension=32)
    with pytest.raises(ValueError):
        k.apply(op, [3, 16])


def test_kernel_rejects_wrong_message_count():
    k = _kernel()
    op = LatticeAirCompOperator(num_nodes=3, modulus=16, dimension=32)
    with pytest.raises(ValueError):
        k.apply(op, [1, 2])


def test_kernel_output_record_shape():
    k = _kernel()
    op = LatticeAirCompOperator(num_nodes=2, modulus=16, dimension=32)
    out = k.apply(op, [1, 2])
    for key in ['value', 'true_value', 'exact', 'messages', 'modulus',
                'coefficients']:
        assert key in out
    assert out['modulus'] == 16
    assert out['coefficients'] == [1, 1]


def test_kernel_result_is_deterministic():
    # Noise-free path: the decoded sum is exact regardless of the internal
    # randomness (which shapes the codewords, not the result).
    k = _kernel()
    ws = [6, 13]
    op = LatticeAirCompOperator(num_nodes=2, modulus=16, dimension=32)
    v1 = k.apply(op, ws)['value']
    v2 = k.apply(op, ws)['value']
    assert v1 == v2 == (6 + 13) % 16


def test_lattice_operator_registered_in_all():
    import rf_compute as pkg
    assert 'LatticeAirCompOperator' in pkg.__all__