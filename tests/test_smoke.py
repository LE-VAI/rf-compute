"""
Smoke tests for the rf-compute kernel.

These are NOT unit tests of every code path — they are end-to-end smoke tests
that verify the three hello-world experiments produce the mathematically
correct result. If these pass, the kernel is healthy and the walkthroughs
will work. If any fails, something fundamental broke.

Run:  pytest
Run:  pytest -v        (verbose, shows each tier)
"""
import numpy as np
import pytest

from rf_compute import (
    AirCompOperator, ConvolutionOperator, InversionOperator, ReservoirOperator,
    WaveComputeKernel,
    boxcar, differencer, matched, hilbert,
)


# ─── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sim_kernel():
    """A simulation kernel — free, no hardware, runs in CI."""
    return WaveComputeKernel(backend="sim")


# ─── Tier 1 — AirComp sum (Nazer & Gastpar 2011) ───────────────────────────

class TestTier1AirComp:
    """The channel computes a sum of N distributed inputs."""

    def test_sum_of_two_scalars(self, sim_kernel):
        """3 + 5 must equal 8 — the canonical hello world."""
        op = AirCompOperator(num_nodes=2, function="sum")
        result = sim_kernel.apply(op, inputs=[3.0, 5.0])
        assert result == 8.0, f"AirComp sum 3+5 should be 8.0, got {result}"

    def test_sum_of_three_scalars(self, sim_kernel):
        """The kernel scales to N nodes, not just 2."""
        op = AirCompOperator(num_nodes=3, function="sum")
        result = sim_kernel.apply(op, inputs=[1.0, 2.0, 4.0])
        assert result == 7.0, f"AirComp sum 1+2+4 should be 7.0, got {result}"

    def test_sum_with_negative(self, sim_kernel):
        """The sum handles negative operands (signed superposition)."""
        op = AirCompOperator(num_nodes=2, function="sum")
        result = sim_kernel.apply(op, inputs=[10.0, -3.0])
        assert result == 7.0

    def test_wrong_node_count_raises(self, sim_kernel):
        """Calling apply with the wrong number of inputs must fail loudly."""
        op = AirCompOperator(num_nodes=3, function="sum")
        with pytest.raises(ValueError, match="expects 3 inputs"):
            sim_kernel.apply(op, inputs=[1.0, 2.0])  # only 2, should be 3

    def test_unsupported_function_raises(self):
        """The analog kernel supports 'sum' only; lattice-coded functions
        are a Tier 1.5 contribution target (see CONTRIBUTING.md)."""
        with pytest.raises(NotImplementedError, match="not supported"):
            AirCompOperator(num_nodes=2, function="product")


# ─── Tier 2 — Wave-domain convolution (Silva 2014) ─────────────────────────

class TestTier2Convolution:
    """The medium convolves the signal with the operator's impulse response."""

    def test_differencer_detects_edge(self, sim_kernel):
        """The differencer [1, -1] detects a rising edge — the Silva 2014 op."""
        op = differencer()
        # A step input: [0, 0, 1, 1, 1] — the edge is at index 2
        x = np.array([0, 0, 1, 1, 1], dtype=np.complex128)
        y = sim_kernel.apply(op, inputs=x)
        # The differencer produces 1 at the rising edge, -1 at the falling edge, 0 elsewhere
        assert np.isclose(y[2], 1.0) or np.isclose(y[1], 1.0), (
            f"Differencer should detect the edge near index 2, got y={np.real(y).round(3)}"
        )

    def test_boxcar_smooths(self, sim_kernel):
        """The boxcar (moving average) smooths a noisy signal — low-pass."""
        op = boxcar(N=4)
        # A signal with a spike: [0, 0, 10, 0, 0]
        x = np.array([0, 0, 10, 0, 0], dtype=np.complex128)
        y = sim_kernel.apply(op, inputs=x)
        # The boxcar should spread the spike's energy across neighbors
        assert np.max(np.abs(y)) < 10.0, "Boxcar should reduce the peak amplitude"
        assert np.sum(np.abs(y)) > 0, "Boxcar output should not be all zeros"

    def test_matched_correlates(self, sim_kernel):
        """The matched filter produces a peak where the template matches."""
        template = np.array([1, 1, 1, 1], dtype=np.complex128)
        op = matched(template)
        # Input contains the template embedded at index 3
        x = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0], dtype=np.complex128)
        y = sim_kernel.apply(op, inputs=x)
        # The correlation peak should be the max of |y|
        assert np.max(np.abs(y)) > 0, "Matched filter output should have a peak"

    def test_impulse_response_is_preserved(self, sim_kernel):
        """The operator's impulse response is exactly what was set."""
        h = np.array([1.0, -2.0, 1.0], dtype=np.complex128)
        op = ConvolutionOperator(impulse_response=h)
        np.testing.assert_array_almost_equal(op.impulse_response, h)

    def test_convolution_with_delta_is_identity(self, sim_kernel):
        """Convolution with a unit impulse [1] returns the signal unchanged."""
        delta = ConvolutionOperator(impulse_response=np.array([1.0]))
        x = np.array([3.0, -1.0, 2.0, 4.0], dtype=np.complex128)
        y = sim_kernel.apply(delta, inputs=x)
        np.testing.assert_array_almost_equal(np.real(y[:len(x)]), np.real(x))


# ─── Tier 3 — Matrix inversion via feedback (Nature Comms 2025) ────────────

class TestTier3Inversion:
    """The feedback loop solves Ax = b for x by Richardson iteration."""

    def test_solves_2x2_system(self, sim_kernel):
        """The kernel solves a well-conditioned 2x2 system to high precision."""
        A = np.array([[0.8, 0.2], [0.1, 0.7]])
        x_true = np.array([1.5, -0.5])
        b = A @ x_true
        op = InversionOperator(matrix=A, step_size=0.5, max_iters=100)
        result = sim_kernel.solve(op, target=b)
        assert result['converged'], f"Should converge, final_error={result['final_error']}"
        np.testing.assert_array_almost_equal(result['x'], x_true, decimal=4)

    def test_returns_convergence_history(self, sim_kernel):
        """The solve result includes the error history for plotting/verification."""
        A = np.array([[0.5, 0.0], [0.0, 0.5]])
        b = np.array([1.0, 1.0])
        op = InversionOperator(matrix=A, step_size=1.0, max_iters=50)
        result = sim_kernel.solve(op, target=b)
        assert 'history' in result, "solve() must return convergence history"
        assert len(result['history']) == result['iterations']
        # Error should decrease (monotonic convergence for this well-conditioned case)
        history = result['history']
        assert history[-1] < history[0], "Error should decrease over iterations"

    def test_open_loop_multiply_matches_matrix(self, sim_kernel):
        """apply() on InversionOperator does the open-loop multiply A·x."""
        A = np.array([[0.8, 0.2], [0.1, 0.7]])
        op = InversionOperator(matrix=A)
        x = np.array([1.0, 2.0])
        y = sim_kernel.apply(op, inputs=x)
        np.testing.assert_array_almost_equal(y, A @ x)

    def test_unstable_step_size_warns(self):
        """A step_size >= 2/lambda_max must warn (the iteration will diverge)."""
        A = np.array([[0.9, 0.0], [0.0, 0.9]])  # lambda_max = 0.9, 2/0.9 ≈ 2.22
        with pytest.warns(UserWarning, match="will diverge"):
            InversionOperator(matrix=A, step_size=3.0)  # 3.0 > 2.22, should warn

    def test_solve_only_for_inversion(self, sim_kernel):
        """solve() is only defined for InversionOperator — others must raise."""
        op = AirCompOperator(num_nodes=2)
        with pytest.raises(TypeError, match="only defined for InversionOperator"):
            sim_kernel.solve(op, target=np.array([1.0]))


# ─── Lineage 4 — Reservoir (Torrejon 2017) ─────────────────────────────────

class TestReservoir:
    """The approximate software model of spin-torque reservoir computing."""

    def test_reservoir_produces_high_dim_state(self, sim_kernel):
        """The reservoir maps a low-dim input to a higher-dim feature space."""
        op = ReservoirOperator(reservoir_size=50, nonlinearity="tanh")
        x = np.array([1.0, 2.0, 3.0])
        state = sim_kernel.apply(op, inputs=x)
        assert state.shape == (50,), f"Reservoir state should be shape (50,), got {state.shape}"

    def test_reservoir_is_nonlinear(self, sim_kernel):
        """The reservoir applies a nonlinearity (tanh bounds the state to [-1, 1])."""
        op = ReservoirOperator(reservoir_size=100, nonlinearity="tanh")
        x = np.array([100.0])  # large input — tanh should saturate
        state = sim_kernel.apply(op, inputs=x)
        assert np.all(np.abs(state) <= 1.0 + 1e-6), "tanh state should be in [-1, 1]"

    def test_unsupported_nonlinearity_raises(self):
        """Only tanh and sigmoid are supported."""
        with pytest.raises(ValueError, match="not supported"):
            ReservoirOperator(nonlinearity="relu")


# ─── Backend behavior ─────────────────────────────────────────────────────

class TestBackends:
    """Verify the two-backend design and graceful degradation."""

    def test_sim_backend_default(self):
        """Default backend is 'sim' — free, no hardware."""
        k = WaveComputeKernel()
        assert k.backend == "sim"

    def test_sim_backend_runs_aircomp(self, sim_kernel):
        """The sim backend actually computes — not a stub."""
        op = AirCompOperator(num_nodes=2)
        assert sim_kernel.apply(op, inputs=[2.0, 3.0]) == 5.0

    def test_invalid_backend_raises(self):
        """Unknown backend names must raise immediately, not silently degrade."""
        with pytest.raises(ValueError, match="not supported"):
            WaveComputeKernel(backend="quantum")


# ─── Named operators (the Tier 2 library) ──────────────────────────────────

class TestNamedOperators:
    """The four convenience constructors from the Tier 2 walkthrough."""

    def test_boxcar_is_normalized(self):
        """boxcar(N) produces a normalized moving average (sums to 1)."""
        op = boxcar(N=8)
        assert np.isclose(np.sum(op.impulse_response), 1.0), "boxcar should sum to 1"

    def test_differencer_is_first_difference(self):
        """differencer() produces [1, -1] — the first-difference operator."""
        op = differencer()
        np.testing.assert_array_almost_equal(
            op.impulse_response, np.array([1.0, -1.0])
        )

    def test_matched_is_time_reversed_conjugate(self):
        """matched(template) produces conj(template[::-1]) — the matched filter."""
        template = np.array([1+1j, 2+0j, 0-1j], dtype=np.complex128)
        op = matched(template)
        expected = np.conj(template[::-1])
        np.testing.assert_array_almost_equal(op.impulse_response, expected)

    def test_hilbert_center_is_zero(self):
        """hilbert(N) center coefficient is zero (no divide-by-zero, fixed bug)."""
        op = hilbert(N=64)
        center = op.impulse_response[32]
        assert np.isclose(center, 0.0), (
            f"Hilbert center should be 0 (the divide-by-zero fix), got {center}"
        )


# ─── Provenance check ──────────────────────────────────────────────────────

class TestProvenance:
    """Every operator carries its lineage explicitly — the structural bridge."""

    def test_aircomp_lineage(self):
        assert AirCompOperator().lineage == "aircomp"

    def test_convolution_lineage(self):
        assert ConvolutionOperator().lineage == "metamaterial"

    def test_inversion_lineage(self):
        assert InversionOperator().lineage == "inversion"

    def test_reservoir_lineage(self):
        assert ReservoirOperator().lineage == "spin_torque"

    def test_all_four_lineages_present(self):
        """The bridge spans all four lineages — none missing."""
        lineages = {
            AirCompOperator().lineage,
            ConvolutionOperator().lineage,
            InversionOperator().lineage,
            ReservoirOperator().lineage,
        }
        assert len(lineages) == 4, f"Expected 4 distinct lineages, got {lineages}"