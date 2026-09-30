"""
Tier 4 — over-the-air federated learning.

Each test pins one claim the walkthrough makes
(docs/hello-world-ota-federated-learning.md), so a change that breaks the
physics or the scoreline fails here first.
"""
import numpy as np
import pytest

from rf_compute import OTAAggregationOperator, WaveComputeKernel
from rf_compute import ota_fl
from rf_compute.ota_fl import (
    SCHEMES, aggregation_quality, gradient_spread, make_federated_data,
    ota_aggregate, scoreline, train,
)


def _median_nmse(grads, trials=150, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return float(np.median([ota_aggregate(grads, rng=rng, **kw)['nmse'] for _ in range(trials)]))


def _similar_grads(spread, k=20, d=20, seed=1):
    """k gradients around a common direction; spread scales the per-device deviation."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=d)
    base *= 0.5 / np.linalg.norm(base)
    return [base + spread * 0.5 * rng.normal(size=d) / np.sqrt(d) for _ in range(k)]


# ─── One aggregation: the wave domain ──────────────────────────────────────

class TestAggregation:

    def test_aligned_channel_recovers_the_average(self):
        """No misalignment, high SNR: the superposition IS the average."""
        grads = _similar_grads(1.0)
        assert _median_nmse(grads, snr_db=40.0, receiver='naive') < 1e-3

    def test_transmit_power_never_exceeds_budget(self):
        """Truncated channel inversion keeps every participant within P = 1."""
        rng = np.random.default_rng(3)
        for _ in range(200):
            grads = [rng.normal(size=12) * rng.uniform(0.1, 3.0) for _ in range(10)]
            out = ota_aggregate(grads, rng=rng, max_phase=np.pi, max_timing=1.0)
            assert out['max_tx_power'] <= 1.0 + 1e-9

    def test_truncation_sets_participation(self):
        """Devices below the truncation threshold sit the round out; 0 admits all."""
        grads = _similar_grads(1.0)
        rng = np.random.default_rng(4)
        assert ota_aggregate(grads, rng=rng, truncation=0.0)['participants'] == len(grads)
        parts = [ota_aggregate(grads, rng=rng, truncation=1.0)['participants'] for _ in range(300)]
        # P(|h| >= 1) = exp(-1) for h ~ CN(0, 1)
        assert abs(np.mean(parts) / len(grads) - np.exp(-1)) < 0.03

    def test_channel_uses_do_not_scale_with_devices(self):
        """d + 1 samples plus the pilot, whether 5 devices transmit or 50."""
        few = ota_aggregate(_similar_grads(1.0, k=5), rng=np.random.default_rng(0))
        many = ota_aggregate(_similar_grads(1.0, k=50), rng=np.random.default_rng(0))
        assert few['channel_uses'] == many['channel_uses'] == 20 + 1 + 32

    def test_equalizer_is_near_exact_when_devices_agree(self):
        """Identical gradients: the aggregate taps describe every device exactly."""
        grads = _similar_grads(0.0)
        kw = dict(snr_db=30.0, max_phase=np.pi / 3, max_timing=0.5)
        assert _median_nmse(grads, receiver='equalized', **kw) < 0.01
        assert _median_nmse(grads, receiver='naive', **kw) > 0.1

    def test_equalizer_error_grows_with_heterogeneity(self):
        """The residual is the device-to-device difference, smeared by the offsets."""
        kw = dict(snr_db=30.0, max_phase=np.pi / 3, max_timing=0.5, receiver='equalized')
        low = _median_nmse(_similar_grads(0.2), **kw)
        high = _median_nmse(_similar_grads(3.0), **kw)
        assert high > 3 * low

    def test_no_phase_sync_makes_naive_useless(self):
        """phi ~ U(-pi, pi): E[cos phi] = 0, so the naive estimate is no better than zero."""
        grads = _similar_grads(0.5)
        nmse = _median_nmse(grads, snr_db=30.0, max_phase=np.pi, max_timing=1.0, receiver='naive')
        assert nmse > 0.8

    def test_bad_receiver_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            ota_aggregate(_similar_grads(1.0), receiver='oracle')


# ─── The learning problem ──────────────────────────────────────────────────

class TestData:

    def test_heterogeneity_knob_spreads_the_gradients(self):
        spreads = [gradient_spread(make_federated_data(heterogeneity=h, rng=np.random.default_rng(7))[0])
                   for h in (0.0, 2.0, 4.0)]
        assert spreads[0] < spreads[1] < spreads[2]

    def test_orthogonal_training_learns(self):
        devices, test, _ = make_federated_data(rng=np.random.default_rng(0))
        hist = train('orthogonal', devices=devices, test=test, rounds=40,
                     rng=np.random.default_rng(1))
        assert hist['test_accuracy'][-1] > 0.8
        assert hist['test_loss'][-1] < hist['test_loss'][0]

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            train('carrier_pigeon', rounds=1)


# ─── The scoreline ─────────────────────────────────────────────────────────

# Module-level so it works on every supported Python: a class-scoped fixture
# must be a plain function (pytest 10 removes the instance-method form), and
# `@staticmethod` is not a substitute -- staticmethod objects only gained
# __name__ in 3.10, which breaks collection on 3.9.
@pytest.fixture(scope="module")
def severe():
    return scoreline(rounds=40, num_seeds=4, seed=0)


class TestScoreline:

    def test_equalizer_matches_orthogonal_without_phase_sync(self, severe):
        assert abs(severe['ota_equalized']['accuracy_mean']
                   - severe['orthogonal']['accuracy_mean']) < 0.02

    def test_naive_receiver_is_a_coin_flip_without_phase_sync(self, severe):
        assert severe['ota_misaligned']['accuracy_mean'] < 0.7
        assert severe['orthogonal']['accuracy_min'] > 0.8

    def test_over_the_air_uses_fewer_channel_uses(self, severe):
        # 20 devices x 20 features per round orthogonal, vs 20 + 1 + 32 over the air
        assert severe['orthogonal']['channel_uses'] == 40 * 20 * 20
        assert severe['ota_equalized']['channel_uses'] == 40 * (20 + 1 + 32)

    def test_moderate_misalignment_does_not_stop_learning(self):
        """A consistent shrink acts like a smaller step: aggregation error, same accuracy."""
        s = scoreline(rounds=40, num_seeds=3, seed=0, max_phase=np.pi / 3, max_timing=0.5)
        assert s['ota_misaligned']['aggregation_nmse'] > 0.1
        assert abs(s['ota_misaligned']['accuracy_mean'] - s['orthogonal']['accuracy_mean']) < 0.02

    def test_seed_deterministic(self):
        a = scoreline(rounds=10, num_seeds=2, seed=5)
        b = scoreline(rounds=10, num_seeds=2, seed=5)
        for k in SCHEMES:
            assert a[k] == b[k]


# ─── The kernel surface ────────────────────────────────────────────────────

class TestKernelSurface:

    def test_kernel_applies_ota_aggregation(self):
        kernel = WaveComputeKernel(backend="sim")
        op = OTAAggregationOperator(snr_db=40.0, seed=2)
        out = kernel.apply(op, inputs=_similar_grads(1.0))
        assert out['estimate'].shape == (20,)
        assert out['nmse'] < 0.01

    def test_operator_validates_receiver_and_timing(self):
        with pytest.raises(ValueError, match="not supported"):
            OTAAggregationOperator(receiver='oracle')
        with pytest.raises(ValueError, match="max_timing"):
            OTAAggregationOperator(max_timing=1.5)

    def test_lineage_is_aircomp(self):
        assert OTAAggregationOperator().lineage == "aircomp_fl"
