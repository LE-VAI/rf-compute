"""
Hardware reality layer — the impairments the Tier 4 model leaves out.

Each test pins one claim the module docstring makes. The load-bearing ones:

  - the documented ESP32 10-bit I/Q word format round-trips
  - the ideal configuration IS the Tier 4 model (so the layer is additive,
    not a rival model)
  - the naive receiver dies under a real receiver, as it does in Tier 4
  - CFO — not quantization, not IQ imbalance — is what kills the equalizer
  - a COMMON clock is worse than independent offsets, because coherent error
    adds instead of averaging
  - the equalizer's ceiling is an accumulated-phase budget, not a tuning knob

The numbers are asserted as orderings and ratios rather than magic constants,
so a change that breaks the physics fails here while a change that merely
reshuffles RNG draws does not.
"""
import numpy as np
import pytest

from rf_compute import hardware as hw
from rf_compute import ota_fl
from rf_compute.hardware import (
    DEV_BOARD_CLOCKED, DEV_BOARD_UNCALIBRATED, PHONE_GRADE,
    HardwareImpairments, esp_iq_word_layout, ota_aggregate_hardware,
    overhead_table, pack_esp_iq, survival_table, unpack_esp_iq,
)


# ─── fixtures ──────────────────────────────────────────────────────────────

def _grads(heterogeneity=2.0, seed=7):
    devices, _, _ = ota_fl.make_federated_data(
        num_devices=20, dim=20, heterogeneity=heterogeneity,
        rng=np.random.default_rng(seed))
    d = devices[0][0].shape[1]
    return [ota_fl.local_gradient(np.zeros(d), X, y) for X, y in devices]


def _median_nmse(grads, cfg, *, rx='equalized', sub=1, pilot=32, trials=80,
                 seed=50, **kw):
    vals = [ota_aggregate_hardware(
        grads, cfg=cfg, snr_db=20.0, rng=np.random.default_rng(seed + i),
        truncation=0.3, clip=1.0, max_phase=kw.pop('max_phase', np.pi),
        max_timing=kw.pop('max_timing', 0.5), receiver=rx,
        sub_blocks=sub, pilot_len=pilot, **kw)['nmse'] for i in range(trials)]
    return float(np.median(vals))


# ─── the documented capture format ──────────────────────────────────────────

class TestEspIqFormat:

    def test_layout_matches_the_documented_shifts(self):
        """The word layout is the format the disclosure documents."""
        lay = esp_iq_word_layout()
        assert lay['word_bits'] == 32
        assert lay['i'] == {'shift': 10, 'bits': 10, 'signed': True}
        assert lay['q'] == {'shift': 0, 'bits': 10, 'signed': True}
        assert lay['gain']['shift'] == 20
        assert lay['agc_fsm']['shift'] == 28

    def test_round_trip_is_exact_for_the_full_10_bit_range(self):
        """Pack then unpack must be lossless, including the sign edges."""
        rng = np.random.default_rng(0)
        i = rng.integers(-512, 512, size=500)
        q = rng.integers(-512, 512, size=500)
        back = unpack_esp_iq(pack_esp_iq(i, q))
        assert np.array_equal(back, i + 1j * q)

    def test_sign_extension_handles_the_minimum(self):
        """-512 is the 10-bit minimum and must not come back as +512."""
        back = unpack_esp_iq(pack_esp_iq(np.array([-512]), np.array([-512])))
        assert back[0] == -512 - 512j

    def test_scale_divides(self):
        w = pack_esp_iq(np.array([256]), np.array([-256]))
        assert unpack_esp_iq(w, scale=512.0)[0] == 0.5 - 0.5j

    def test_gain_bits_do_not_leak_into_iq(self):
        """High nibbles carry gain/AGC state; they must not corrupt I or Q."""
        w = pack_esp_iq(np.array([123]), np.array([-45]), gain=200, agc_fsm=9)
        assert unpack_esp_iq(w)[0] == 123 - 45j


# ─── the layer must be additive, not a rival model ──────────────────────────

class TestIdealLimit:

    def test_ideal_flags_itself(self):
        assert HardwareImpairments().is_ideal
        assert not DEV_BOARD_UNCALIBRATED.is_ideal
        assert not HardwareImpairments(cfo_rad_per_symbol=0.01).is_ideal

    def test_zero_cfo_naive_matches_the_tier4_reference(self):
        """
        With every impairment off and the naive receiver on an aligned channel,
        the layer must agree with Tier 4's own aggregation to within Monte-Carlo
        error. Bitwise equality is impossible by construction — the two
        implementations draw their random numbers in a different order — so the
        claim under test is statistical equivalence.
        """
        grads = _grads()
        rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)
        a = float(np.median([ota_fl.ota_aggregate(
            grads, snr_db=20.0, rng=rng_a, max_phase=0.0, max_timing=0.0,
            receiver='naive')['nmse'] for _ in range(300)]))
        b = float(np.median([ota_aggregate_hardware(
            grads, cfg=HardwareImpairments(), snr_db=20.0, rng=rng_b,
            max_phase=0.0, max_timing=0.0, receiver='naive',
            sub_blocks=1)['nmse'] for _ in range(300)]))
        assert b == pytest.approx(a, rel=0.35)

    def test_no_impairment_returns_a_finite_estimate(self):
        out = ota_aggregate_hardware(_grads(), cfg=HardwareImpairments(),
                                     rng=np.random.default_rng(1),
                                     max_phase=np.pi, max_timing=0.5)
        assert np.all(np.isfinite(out['estimate']))
        assert out['coverage'] == 1.0 and out['nan_windows'] == 0

    def test_transmit_power_stays_within_budget_under_every_preset(self):
        """The impairment layer must not break truncated channel inversion."""
        rng = np.random.default_rng(4)
        for cfg in (PHONE_GRADE, DEV_BOARD_CLOCKED, DEV_BOARD_UNCALIBRATED):
            for _ in range(20):
                out = ota_aggregate_hardware(
                    _grads(seed=int(rng.integers(0, 10_000))), cfg=cfg, rng=rng,
                    max_phase=np.pi, max_timing=0.5)
                assert out['max_tx_power'] <= 1.0 + 1e-9


# ─── what actually kills the receiver ───────────────────────────────────────

class TestImpairmentAttribution:

    def test_cfo_is_the_dominant_impairment(self):
        """
        The headline attribution. A residual carrier offset destroys the
        equalizer; the receiver-chain impairments everyone worries about do not.
        Asserted as an ordering so it survives re-tuning of the presets.
        """
        grads = _grads()
        clean = _median_nmse(grads, HardwareImpairments())
        cfo = _median_nmse(grads, HardwareImpairments(cfo_rad_per_symbol=0.35))
        adc = _median_nmse(grads, HardwareImpairments(adc_bits=10))
        iq = _median_nmse(grads, HardwareImpairments(iq_imbalance_db=-30.0))
        agc = _median_nmse(grads, HardwareImpairments(agc_gain_error_db=1.0))
        assert cfo > 5.0 * clean
        for benign in (adc, iq, agc):
            assert benign == pytest.approx(clean, rel=0.25)

    def test_burst_gaps_are_the_second_order_impairment(self):
        """Missing capture windows hurt, and hurt more than the ADC path does."""
        grads = _grads()
        clean = _median_nmse(grads, HardwareImpairments())
        burst = _median_nmse(grads, HardwareImpairments(duty_cycle=0.6))
        adc = _median_nmse(grads, HardwareImpairments(adc_bits=10))
        assert burst > 3.0 * clean
        assert burst > adc

    def test_duty_cycle_below_one_reports_missing_windows(self):
        """A gap must be visible as missing windows, not silently interpolated."""
        out = ota_aggregate_hardware(
            _grads(), cfg=HardwareImpairments(duty_cycle=0.5),
            rng=np.random.default_rng(2), max_phase=np.pi, max_timing=0.5)
        assert out['nan_windows'] > 0
        assert 0.4 < out['coverage'] < 0.65

    def test_nan_windows_never_reach_the_estimate(self):
        """Gaps are zero-filled, so the returned estimate stays finite."""
        out = ota_aggregate_hardware(
            _grads(), cfg=HardwareImpairments(duty_cycle=0.5, gap_model='scatter'),
            rng=np.random.default_rng(3), max_phase=np.pi, max_timing=0.5)
        assert np.all(np.isfinite(out['estimate']))


# ─── the counterintuitive result ────────────────────────────────────────────

class TestSynchronizationThesis:

    def test_common_clock_is_not_a_free_rescue(self):
        """
        The result that inverts the intuition.

        One might assume a shared clock fixes the equalizer. It does not: a CFO
        that is COMMON to every device produces errors that add COHERENTLY, so it
        can be worse than independent offsets, which partially average out. This
        is why 'just synchronize the clocks' is not sufficient — the sync target
        must be zero residual offset, not merely a common one.
        """
        grads = _grads()
        indep = _median_nmse(grads, HardwareImpairments(cfo_rad_per_symbol=0.35))
        common = _median_nmse(grads, HardwareImpairments(cfo_common_rad_per_symbol=0.35))
        assert common > indep

    def test_equalizer_beats_naive_only_while_cfo_is_small(self):
        """The equalizer is not universally better — it is better inside a budget."""
        grads = _grads(heterogeneity=2.0)
        ok = _median_nmse(grads, HardwareImpairments(cfo_rad_per_symbol=0.01))
        bad = _median_nmse(grads, HardwareImpairments(cfo_rad_per_symbol=0.35))
        assert ok < 1.0
        assert bad > 1.0


# ─── the ceiling, expressed as an engineering budget ────────────────────────

class TestAccumulatedPhaseCeiling:

    def test_equalizer_holds_under_half_a_radian_accumulated(self):
        """
        The actionable engineering number. Sweep the CFO so that the phase
        accumulated across the data block crosses roughly half a radian, and
        assert the equalizer holds below it and fails above it. This is the
        budget a sync stage has to hit.
        """
        grads = _grads()

        def nmse_over_block(cfo, d=20):
            cfg = HardwareImpairments(cfo_rad_per_symbol=cfo)
            return _median_nmse(grads, cfg), cfo * d

        inside, accum_in = nmse_over_block(0.02)     # 0.40 rad over the block
        outside, accum_out = nmse_over_block(0.08)   # 1.60 rad over the block
        assert accum_in < 0.6 and inside < 0.6
        assert accum_out > 1.5 and outside > 1.5

    def test_monotone_in_cfo(self):
        """More residual carrier offset must never improve the estimate."""
        grads = _grads()
        vals = [_median_nmse(grads, HardwareImpairments(cfo_rad_per_symbol=c),
                             trials=60) for c in (0.0, 0.02, 0.08, 0.35)]
        assert vals == sorted(vals)

    def test_tracking_does_not_rescue_a_severe_offset(self):
        """
        Midambles re-measured close to the data do not save a receiver whose
        coherence time is a fraction of a symbol. It is an architecture ceiling,
        not a missing parameter — and this test exists so nobody re-derives it
        and thinks they found the fix.
        """
        grads = _grads()
        cfg = HardwareImpairments(cfo_rad_per_symbol=0.35)
        one = _median_nmse(grads, cfg, sub=1, pilot=8, trials=60)
        many = _median_nmse(grads, cfg, sub=8, pilot=8, trials=60)
        assert many > 1.0
        assert one > 1.0


# ─── the price of the fix, and the tables ───────────────────────────────────

class TestOverhead:

    def test_more_midambles_cost_channel_uses(self):
        """Overhead must be monotone in the sub-block count, or the trade is fake."""
        devices, _, _ = ota_fl.make_federated_data(
            num_devices=20, dim=20, heterogeneity=2.0,
            rng=np.random.default_rng(7))
        table = overhead_table(devices, cfg=DEV_BOARD_CLOCKED,
                                sub_block_counts=(1, 2, 4), trials=20, seed=1)
        assert table[1]['channel_uses'] < table[2]['channel_uses'] < table[4]['channel_uses']
        assert table[1]['pilot_overhead'] < table[4]['pilot_overhead']

    def test_survival_table_has_every_receiver_column(self):
        devices, _, _ = ota_fl.make_federated_data(
            num_devices=20, dim=20, heterogeneity=2.0,
            rng=np.random.default_rng(7))
        table = survival_table(devices, trials=20, seed=1)
        assert 'ideal (Tier 4 model)' in table
        for cells in table.values():
            for col in ('aligned', 'naive', 'equalized'):
                assert col in cells
                assert np.isfinite(cells[col])


# ─── the tracking receiver ───────────────────────────────────────────────────

class TestTrackingReceiver:
    """
    The receiver that answers the question Tier 4-h raised. Tier 4-h established
    that a block-fit equalizer cannot follow a rotating phase; this pins what a
    properly tuned tracking loop CAN do, and — just as important — what it still
    cannot.
    """

    def test_tracking_holds_with_no_impairment(self):
        """The loop must not be worse than the block fit when nothing is wrong."""
        grads = _grads()
        track = _median_nmse(grads, HardwareImpairments(), rx='tracking', trials=60)
        assert track < 0.75

    def test_tracking_rescues_a_common_rotation(self):
        """
        The result that matters. A rotation SHARED by every device is one phase
        trajectory, and a second-order loop follows it: the block fit degrades to
        ~22 at 0.35 rad/symbol while the tracker holds under 0.75.
        """
        grads = _grads()
        cfg = HardwareImpairments(cfo_common_rad_per_symbol=0.35)
        eq = _median_nmse(grads, cfg, rx='equalized', trials=80)
        tr = _median_nmse(grads, cfg, rx='tracking', trials=80)
        assert eq > 5.0
        assert tr < 0.75

    def test_tracking_beats_block_fit_beyond_the_half_radian_wall(self):
        """
        Tier 4-h found a block-fit ceiling at ~0.5 rad accumulated per block. The
        tracker clears 7 rad. Both sides asserted, so a change that breaks either
        the ceiling or the rescue fails here.
        """
        grads = _grads()
        cfg = HardwareImpairments(cfo_common_rad_per_symbol=0.35)   # 7 rad / block
        eq = _median_nmse(grads, cfg, rx='equalized', trials=60)
        tr = _median_nmse(grads, cfg, rx='tracking', trials=60)
        assert eq > 5.0 * tr

    def test_loop_gain_is_a_real_parameter_not_a_formality(self):
        """
        A narrow loop fails where a tuned one holds. Recorded because 'add a
        tracking loop' is true only of a properly tuned loop — a narrow one is the
        same failure in a different costume.
        """
        grads = _grads()
        cfg = HardwareImpairments(cfo_common_rad_per_symbol=0.20)
        narrow = _median_nmse(grads, cfg, rx='tracking', trials=60,
                              pll_kp=0.25, pll_ki=0.01)
        tuned = _median_nmse(grads, cfg, rx='tracking', trials=60,
                             pll_kp=0.8, pll_ki=0.2)
        assert tuned < narrow
        assert tuned < 0.75

    def test_tracking_cannot_rescue_independent_offsets(self):
        """
        The structural limit, and the most useful negative in this file.

        A single-phase tracker follows ONE rotation. Independent per-device offsets
        give every device its own, so the tracked phase is a compromise rather than
        a fix. The genie tests below measure how much of that is the receiver
        and how much the channel. This is why the sync target is zero residual
        offset, not merely a small one.
        """
        grads = _grads()
        cfg = HardwareImpairments(cfo_rad_per_symbol=0.35)
        tr = _median_nmse(grads, cfg, rx='tracking', trials=60)
        assert tr > 1.0

    def test_genie_most_of_the_independent_offset_gap_is_the_receiver(self):
        """
        Is the independent-offset limit the receiver's or the channel's?

        The genie is handed the true combined channel at every symbol, so it
        bounds every receiver that models the sum's channel. Under independent
        offsets it stays useful (NMSE < 1) and far below the single-phase
        tracker: most of the tracker's failure is the receiver.

        (This replaces an earlier coherence test, |sum_k e^{i theta_k}| / K. With
        start phases uniform on (-pi, pi) that magnitude sits near
        sqrt(pi / 4K) for ANY phase trajectory, including phases re-drawn every
        symbol, so it could not tell a receiver limit from a channel limit.)
        """
        grads = _grads()
        cfg = HardwareImpairments(cfo_rad_per_symbol=0.35)
        genie = _median_nmse(grads, cfg, rx='genie', trials=80)
        tr = _median_nmse(grads, cfg, rx='tracking', trials=80)
        assert genie < 1.0
        assert tr > 3 * genie

    def test_genie_independent_offsets_still_cost_something(self):
        """
        ...and some of it is the channel. Knowing the combined channel perfectly,
        a common rotation costs nothing, but a spread of independent offsets
        still raises the error: device-to-device differences are smeared by
        phases that now move. 'Mostly the receiver', not 'only the receiver'.
        """
        grads = _grads()
        clean = _median_nmse(grads, HardwareImpairments(), rx='genie', trials=80)
        common = _median_nmse(grads, HardwareImpairments(cfo_common_rad_per_symbol=0.35),
                              rx='genie', trials=80)
        spread = _median_nmse(grads, HardwareImpairments(cfo_rad_per_symbol=0.35),
                              rx='genie', trials=80)
        assert abs(common - clean) < 0.25 * clean
        assert spread > 1.3 * clean

    def test_tracking_is_finite_and_correctly_shaped(self):
        """No NaN, no refusal to produce output, right length."""
        out = ota_aggregate_hardware(
            _grads(), cfg=DEV_BOARD_UNCALIBRATED, rng=np.random.default_rng(9),
            max_phase=np.pi, max_timing=0.5, receiver='tracking')
        assert out['estimate'].shape == (20,)
        assert np.all(np.isfinite(out['estimate']))


# ─── guards ─────────────────────────────────────────────────────────────────

class TestGuards:

    def test_rejects_impossible_duty_cycle(self):
        with pytest.raises(ValueError):
            HardwareImpairments(duty_cycle=0.0)
        with pytest.raises(ValueError):
            HardwareImpairments(duty_cycle=1.5)

    def test_rejects_unknown_gap_model(self):
        with pytest.raises(ValueError):
            HardwareImpairments(gap_model='clustered')

    def test_rejects_negative_adc_bits(self):
        with pytest.raises(ValueError):
            HardwareImpairments(adc_bits=-1)

    def test_rejects_unknown_receiver(self):
        with pytest.raises(ValueError):
            ota_aggregate_hardware(_grads(), cfg=HardwareImpairments(),
                                   rng=np.random.default_rng(0), receiver='ml')

    def test_rejects_zero_sub_blocks(self):
        with pytest.raises(ValueError):
            ota_aggregate_hardware(_grads(), cfg=HardwareImpairments(),
                                   rng=np.random.default_rng(0), sub_blocks=0)

    def test_presets_are_stated_assumptions_and_labelled_as_such(self):
        """The presets are not measurements; the docstring must say so."""
        doc = (hw.__doc__ or '') + (HardwareImpairments.__doc__ or '')
        assert 'NOT measurements' in doc or 'not measurements' in doc
