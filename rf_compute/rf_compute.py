"""
rf-compute: wave-domain computation kernel
==========================================

The unified developer surface for RF-as-compute. One API across all four
lineages: computational metamaterials, over-the-air computation (AirComp),
microwave photonics, and spin-torque neuromorphic RF.

The kernel's job: let a builder write the same code whether the operator is
an AirComp sum, a metamaterial convolution, or a feedback-loop inversion.
The lineage is explicit in the operator type (one class per lineage), so the
bridge is structural, not notional — each lineage keeps its physics, the
kernel maps them to a common interface.

Two backends:
  - "sim"   — pure NumPy, no hardware. Free. Run anywhere. The default.
  - "sdr"   — real Software-Defined Radio. Costs $200-$350. Needs SoapySDR.

The SDR backend gracefully degrades: if no SDR is attached, it falls back
to the simulation backend and warns, so code written for hardware runs in
simulation when the hardware is absent.

Quick start:
    from rf_compute import AirCompOperator, ConvolutionOperator, InversionOperator
    from rf_compute import WaveComputeKernel

    kernel = WaveComputeKernel(backend="sim")  # free, no hardware
    op = AirCompOperator(num_nodes=2)
    result = kernel.apply(op, inputs=[3.0, 5.0])  # -> 8.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np

# Backend detection — graceful degradation
try:
    from SoapySDR import Device as _SDRDevice, SOAPY_SDR_TX as _TX, SOAPY_SDR_RX as _RX, SOAPY_SDR_CF32 as _CF32
    _SDR_AVAILABLE = True
except ImportError:
    _SDR_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Operator types — one per lineage
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Operator:
    """
    Base class for wave-domain operators. Each subclass fixes a lineage.

    The lineage is explicit (not hidden in a flag) because the four lineages
    have genuinely different physics. Flattening them into one class would
    erase the structural bridge — a convolution (Lineage 1) is not a sum
    (Lineage 2), even though both are linear operations.
    """
    lineage: str = "base"  # overridden by subclasses
    notes: str = ""        # human-readable scope / provenance note


@dataclass
class AirCompOperator(Operator):
    """
    Lineage 2 — Over-the-Air Computation (analog superposition).

    The wireless channel computes a function (sum, by default) of N
    distributed inputs. The channel is the adder; interference is the
    operation, not noise.

    Maps to: Nazer & Gastpar, IEEE Trans. Inf. Theory (2011).
    Walkthrough: docs/hello-world-aircomp.md

    For the noise-resilient exact version, use LatticeAirCompOperator
    (Tier 1.5) — same lineage, nested-lattice codes.
    """
    num_nodes: int = 2
    function: str = "sum"   # the analog-superposition kernel supports 'sum' only
    lineage: str = "aircomp"

    def __post_init__(self):
        if self.function != "sum":
            raise NotImplementedError(
                f"AirComp function '{self.function}' not supported. "
                "The analog-superposition kernel supports 'sum' only. "
                "For exact finite-field functions use LatticeAirCompOperator "
                "(nested-lattice coding, Tier 1.5)."
            )


@dataclass
class LatticeAirCompOperator(Operator):
    """
    Lineage 2 — Over-the-Air Computation, nested-lattice coded (Tier 1.5).

    The full Nazer/Gastpar result: with nested lattice codes, the channel
    superposition computes an EXACT function of the messages (the integer
    sum mod L) at finite SNR — error probability decaying exponentially in
    the lattice dimension — while the receiver decodes NO individual
    message. One channel use for N nodes, at the single-user rate.

    Maps to: Nazer & Gastpar, IEEE Trans. Inf. Theory 53(10) 3498 (2007,
    founding) and 57(10) 6463 (2011, compute-and-forward,
    DOI 10.1109/TIT.2011.2165816).
    Walkthrough: docs/hello-world-lattice-aircomp.md

    The lattice machinery lives in rf_compute.lattice; this operator binds
    it to the kernel's unified interface.

    coefficients are the compute-and-forward equation: with coefficients
    [a_1..a_N] the decoded value is sum(a_i * w_i) mod L. Integer
    coefficients are realized as channel gains (node i's signal arrives at
    gain a_i — the theory's "channel as the operator" made physical) and
    the receiver replays dithers at those weights.
    """
    num_nodes: int = 2
    modulus: int = 16          # L: the message alphabet Z_L and the coarse cell
    dimension: int = 32        # n: lattice coordinates (error ~ exp decay in n)
    coefficients: Optional[Sequence] = None   # integer a_i; None = all-ones (plain sum)
    lineage: str = "aircomp_lattice"

    def __post_init__(self):
        if self.modulus < 2:
            raise ValueError("modulus must be >= 2")
        if self.dimension < 1:
            raise ValueError("dimension must be >= 1")
        if self.coefficients is not None:
            self.coefficients = [int(c) for c in self.coefficients]
            if len(self.coefficients) != self.num_nodes:
                raise ValueError(
                    f"coefficients length {len(self.coefficients)} != num_nodes {self.num_nodes}"
                )


@dataclass
class FadingAirCompOperator(Operator):
    """
    Lineage 2 — Over-the-Air Computation on a FADING channel (Tier 1.6).

    Tier 1.5 SET the channel gains to integers (h_i = a_i) — exact, but it
    hides the real problem: on a fading channel the receiver must CHOOSE
    which integer combination to decode. This operator runs that choice:

      - gains drawn i.i.d. (real baseband fading),
      - coefficients SELECTED by maximizing the Nazer/Gastpar computation
        rate R = 1/2 log2(1 / (alpha^2/SNR + ||alpha h - a||^2)) over a,
      - both strategies reported: the selected combination vs. the
        transmit-for-plain-sum baseline.

    What the numbers show (see docs/hello-world-fading-coefficients.md):
    selection finds a decodable equation where the plain sum's rate is zero
    (its alignment is wrong for the realized fading) and costs a fraction
    of the power channel inversion would burn on deep fades — selection
    DROPS a faded node (a_i = 0) instead of paying to invert it.

    Maps to: Nazer & Gastpar, IEEE Trans. Inf. Theory 57(10) 6463 (2011);
    coefficient algorithms — Sahraei & Gastpar (Allerton 2014, exact
    polynomial), Liu & Ling, IEEE TWC 15(12) 8039 (2016, efficient search);
    MMSE scaling and the rate formula — Huang & Burr, arXiv:1704.05007.
    Walkthrough: docs/hello-world-fading-coefficients.md
    """
    num_nodes: int = 3
    modulus: int = 16
    dimension: int = 32
    snr_db: float = 5.0
    selection: str = "exhaustive"   # 'exhaustive' | 'lll' | 'rounded'
    coefficients: Optional[Sequence] = None   # None = SELECT from the channel
    lineage: str = "aircomp_fading"

    def __post_init__(self):
        if self.modulus < 2:
            raise ValueError("modulus must be >= 2")
        if self.dimension < 1:
            raise ValueError("dimension must be >= 1")
        if self.selection not in ("exhaustive", "lll", "rounded"):
            raise ValueError(
                f"selection '{self.selection}' not supported; "
                "use 'exhaustive', 'lll', or 'rounded'"
            )
        if self.coefficients is not None:
            self.coefficients = [int(c) for c in self.coefficients]
            if len(self.coefficients) != self.num_nodes:
                raise ValueError(
                    f"coefficients length {len(self.coefficients)} != num_nodes {self.num_nodes}"
                )


@dataclass
class OTAAggregationOperator(Operator):
    """
    Lineage 2 — Over-the-Air Computation as a gradient aggregator (Tier 4).

    K devices transmit their gradient vectors at once over a fading channel
    (truncated channel inversion); the receiver estimates their AVERAGE from
    the superposition. Optional misalignment — residual phase offsets up to
    max_phase and timing offsets up to max_timing symbols — and a choice of
    receiver: 'naive' (reads the superposition as if aligned) or
    'equalized' (a shared pilot measures the aggregate channel taps, then a
    two-tap least-squares deconvolution).

    Maps to: Zhu, Wang & Huang, IEEE TWC 19(1) 491 (2020); misalignment
    model — Shao, Gunduz & Liew, IEEE TWC 21(6) 3951 (2022).
    Walkthrough: docs/hello-world-ota-federated-learning.md
    """
    snr_db: float = 20.0
    max_phase: float = 0.0        # radians; residual phase offsets ~ U(-max_phase, max_phase)
    max_timing: float = 0.0       # symbols; timing offsets ~ U(0, max_timing)
    receiver: str = "equalized"   # 'naive' | 'equalized'
    truncation: float = 0.3       # devices with |h| below this sit the round out
    clip: float = 1.0             # gradient norm clip (bounds transmit power)
    seed: Optional[int] = None    # channel randomness; None draws fresh
    lineage: str = "aircomp_fl"

    def __post_init__(self):
        if self.receiver not in ("naive", "equalized"):
            raise ValueError(
                f"receiver '{self.receiver}' not supported; use 'naive' or 'equalized'"
            )
        if not 0.0 <= self.max_timing <= 1.0:
            raise ValueError("max_timing must be in [0, 1] symbols")


@dataclass
class ConvolutionOperator(Operator):
    """
    Lineage 1 — Computational Metamaterials.

    A linear operator applied to a signal by convolution. The impulse
    response h[n] IS the operator. The medium does the DSP; convolution is
    incurred, not computed.

    Maps to: Silva et al., Science (2014).
    Walkthrough: docs/hello-world-convolution.md
    """
    impulse_response: np.ndarray = field(default_factory=lambda: np.array([1.0, -1.0]))
    lineage: str = "metamaterial"

    def __post_init__(self):
        self.impulse_response = np.asarray(self.impulse_response, dtype=np.complex128)


@dataclass
class InversionOperator(Operator):
    """
    Lineage 1 (closed-loop) — Matrix inversion via feedback.

    Solves Ax = b for x by Richardson iteration with feedback. Open-loop
    does the multiply A·x; closed-loop does the inversion A⁻¹·b. The wave
    domain settles to the solution; computation time = settling time.

    Convergence: the iteration matrix is (I - alpha*A), so the loop settles
    iff its spectral radius rho(I - alpha*A) < 1. For A with real positive
    eigenvalues that is 0 < alpha < 2/lambda_max(A). A's own spectral radius
    is not the condition — A = 2I with alpha = 0.5 converges in one step.

    Maps to: Tzarouchis, Edwards & Engheta, Nature Communications 16, 908
    (2025), DOI 10.1038/s41467-025-56019-1, arXiv:2301.02850.
    Walkthrough: docs/hello-world-matrix-inversion.md
    """
    matrix: np.ndarray = field(default_factory=lambda: np.array([[0.8, 0.2], [0.1, 0.7]]))
    step_size: float = 0.5   # alpha; converges iff rho(I - alpha*A) < 1
    max_iters: int = 50
    lineage: str = "inversion"

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        # Stability check — warn (don't fail) so builders can experiment.
        # Richardson converges iff rho(I - alpha*A) < 1.
        n = self.matrix.shape[0]
        iteration = np.eye(n) - self.step_size * self.matrix
        rho = float(np.max(np.abs(np.linalg.eigvals(iteration))))
        if rho >= 1.0:
            import warnings
            warnings.warn(
                f"rho(I - step_size*A) = {rho:.3f} >= 1.0 with step_size "
                f"{self.step_size}. Iteration will diverge (or stall). For A "
                f"with real positive eigenvalues, use 0 < step_size < 2/lambda_max(A)."
            )


@dataclass
class ReservoirOperator(Operator):
    """
    Lineage 4 — Spin-Torque Neuromorphic RF.

    A nonlinear reservoir computing transform. The RF oscillator's nonlinear
    dynamics map an input signal to a higher-dimensional feature space. The
    "neuron" is a microwave oscillator; computation happens in the dynamics.

    Maps to: Torrejon et al., Nature (2017).
    Simulation note: this is an APPROXIMATE software model of spin-torque
    oscillator dynamics, not a hardware equivalent. The physics (magnetic
    tunnel junctions) is not SDR-reproducible.
    """
    reservoir_size: int = 100
    nonlinearity: str = "tanh"  # activation of the reservoir nodes
    spectral_radius: float = 0.9  # reservoir matrix spectral radius
    lineage: str = "spin_torque"

    def __post_init__(self):
        if self.nonlinearity not in ("tanh", "sigmoid"):
            raise ValueError(f"nonlinearity '{self.nonlinearity}' not supported; use 'tanh' or 'sigmoid'")


# ─────────────────────────────────────────────────────────────────────────────
# The kernel — the unified execution surface
# ─────────────────────────────────────────────────────────────────────────────

class WaveComputeKernel:
    """
    The unified execution surface. One kernel, four lineages, two backends.

    The kernel exposes two operations that span all four lineages:
      - apply(op, inputs)  — apply an operator to inputs (open-loop)
      - solve(op, target)   — solve for the input that produces target (closed-loop, inversion only)

    In simulation backend, apply/solve are pure NumPy. In SDR backend, apply
    transmits inputs at RF and captures the wave-domain output; solve runs
    the feedback loop. The SDR backend degrades to simulation if no hardware.
    """

    def __init__(self, backend: str = "sim", sample_rate: float = 2e6, freq: float = 915e6):
        """
        Args:
            backend:  "sim" (free, default) or "sdr" (real hardware)
            sample_rate: SDR sample rate in Hz (sdr backend only)
            freq: SDR carrier frequency in Hz (sdr backend only; use ISM bands only)
        """
        if backend not in ("sim", "sdr"):
            raise ValueError(f"backend '{backend}' not supported; use 'sim' or 'sdr'")
        self.backend = backend
        self.sample_rate = sample_rate
        self.freq = freq

        # Graceful degradation — SDR backend without hardware falls back to sim
        if self.backend == "sdr" and not _SDR_AVAILABLE:
            import warnings
            warnings.warn(
                "SDR backend requested but SoapySDR not available. "
                "Falling back to simulation. Install SoapySDR + drivers to use real hardware."
            )
            self.backend = "sim"
            self._degraded = True
        else:
            self._degraded = False

        self._tx = None
        self._rx = None

    # ─── public API ───────────────────────────────────────────────────────

    def apply(self, op: Operator, inputs) -> np.ndarray:
        """
        Apply an operator to inputs (open-loop). Works for all four lineages.

        For AirCompOperator:    inputs is a list of N scalar values; returns the sum.
        For OTAAggregationOperator: inputs is a list of K gradient vectors;
                                returns the over-the-air aggregation record.
        For ConvolutionOperator: inputs is a 1-D signal array; returns x * h.
        For ReservoirOperator:  inputs is a 1-D signal (one sample per time
                                step); returns the final reservoir state.
        For InversionOperator:  apply does the open-loop multiply A·x (not the
                                inversion — use solve() for that).
        """
        if isinstance(op, AirCompOperator):
            return self._apply_aircomp(op, inputs)
        elif isinstance(op, LatticeAirCompOperator):
            return self._apply_lattice_aircomp(op, inputs)
        elif isinstance(op, FadingAirCompOperator):
            return self._apply_fading_aircomp(op, inputs)
        elif isinstance(op, OTAAggregationOperator):
            return self._apply_ota_aggregation(op, inputs)
        elif isinstance(op, ConvolutionOperator):
            return self._apply_convolution(op, inputs)
        elif isinstance(op, ReservoirOperator):
            return self._apply_reservoir(op, inputs)
        elif isinstance(op, InversionOperator):
            return self._apply_inversion_openloop(op, inputs)
        else:
            raise TypeError(f"Unknown operator type: {type(op).__name__}")

    def solve(self, op: InversionOperator, target: np.ndarray) -> dict:
        """
        Solve Ax = b for x (closed-loop). Only valid for InversionOperator.

        Returns a dict with:
          - 'x': the solution vector
          - 'history': error at each iteration
          - 'converged': bool
          - 'iterations': count
          - 'final_error': ||Ax - b||

        For the SDR backend, each iteration is a real RF round-trip (the channel
        applies A); the feedback update is computed in software (software-in-
        the-loop). The fully-analog closed loop (Mode C in the walkthrough) is
        a hardware configuration, not a kernel method — see the walkthrough.
        """
        if not isinstance(op, InversionOperator):
            raise TypeError("solve() is only defined for InversionOperator.")
        return self._solve_inversion(op, target)

    # ─── AirComp (Lineage 2) ─────────────────────────────────────────────

    def _apply_aircomp(self, op: AirCompOperator, inputs) -> float:
        inputs = np.asarray(inputs, dtype=np.float64)
        if len(inputs) != op.num_nodes:
            raise ValueError(f"AirCompOperator expects {op.num_nodes} inputs, got {len(inputs)}")
        if op.function == "sum":
            # The wave-domain operation: channel superposition computes the sum.
            # In simulation this is numpy sum; in SDR backend it's real RF
            # superposition (each input transmitted, receiver reads the sum).
            if self.backend == "sdr":
                return self._sdr_aircomp_sum(op, inputs)
            return float(np.sum(inputs))
        raise NotImplementedError(f"function '{op.function}'")  # guarded in __post_init__

    def _sdr_aircomp_sum(self, op: AirCompOperator, inputs) -> float:
        """Real RF AirComp sum — transmits each input, captures superposition."""
        # NOTE: This is a reference SDR path. The full Tier 1 walkthrough
        # (docs/hello-world-aircomp.md) covers channel estimation and
        # pre-coding; this method assumes a calibrated channel.
        raise NotImplementedError(
            "SDR AirComp sum requires channel calibration — see "
            "docs/hello-world-aircomp.md for the full walkthrough. The kernel "
            "method is a stub; the walkthrough is the reference implementation."
        )

    # ─── Lattice AirComp (Lineage 2, Tier 1.5) ────────────────────────────

    def _apply_lattice_aircomp(self, op: LatticeAirCompOperator, inputs) -> dict:
        """
        Run one nested-lattice AirComp round on the INTEGER-ALIGNED channel.

        inputs is a list of N integer messages in {0..L-1}. Coefficients are
        realized as channel gains (h_i = a_i), so this round is exact. Returns:
          'value'         — the decoded sum a·w mod L (exact)
          'true_value'    — ground truth for verification
          'exact'         — bool
          'messages'      — what was transmitted (verification only; the
                            receiver structurally cannot recover these
                            individually from the channel output alone)
          'modulus'       — L
          'coefficients'  — the integer a vector used

        For the FADING channel — gains drawn from a channel, coefficients
        SELECTED by computation-rate maximization rather than set — use
        FadingAirCompOperator, which runs lattice.fading_trial.

        The simulation backend is the reference; the SDR backend transmits the
        same codewords over RF (the walkthrough covers sync and dither-seed
        sharing). SDR lattice transmission degrades to simulation until the
        walkthrough's calibration is implemented in-kernel.
        """
        return self._lattice_round_with_messages(op, inputs, np.random.default_rng())

    def _lattice_round_with_messages(self, op: LatticeAirCompOperator,
                                     ws, rng) -> dict:
        """One lattice AirComp round with caller-supplied messages."""
        from . import lattice as _lattice
        ws = [int(w) for w in ws]
        if len(ws) != op.num_nodes:
            raise ValueError(f"expected {op.num_nodes} messages, got {len(ws)}")
        L = op.modulus
        for w in ws:
            if not (0 <= w < L):
                raise ValueError(f"message {w} out of range [0, {L})")
        a = op.coefficients if op.coefficients is not None else [1] * op.num_nodes
        true_val = int(np.sum(np.asarray(a) * ws) % L)
        xs, ds = [], []
        for w in ws:
            x, d = _lattice.encode(w, L, op.dimension, rng)
            xs.append(x)
            ds.append(d)
        # Noise-free in the sim backend's default: the lattice result is exact.
        # Callers wanting the noisy channel use lattice.channel explicitly.
        # Coefficients are realized as channel GAINS (the compute-and-forward
        # equation made physical): node i's signal arrives at gain a_i, so the
        # superposition is sum(a_i * x_i) and decode — which replays dithers at
        # those same weights — recovers sum(a_i * w_i) mod L exactly.
        y = np.zeros(op.dimension)
        for ai, xi in zip(a, xs):
            y = y + ai * xi
        val = _lattice.decode(y, L, ds, a=a)
        return {
            'value': val,
            'true_value': true_val,
            'exact': val == true_val,
            'messages': ws,
            'modulus': L,
            'coefficients': list(a),
        }

    # ─── Fading AirComp (Lineage 2, Tier 1.6) ─────────────────────────────

    def _apply_fading_aircomp(self, op: FadingAirCompOperator, inputs=None) -> dict:
        """
        Run one fading-channel AirComp round.

        inputs: optional seed (int) or numpy Generator for reproducible
        fading; None draws fresh randomness. The messages, gains, and
        codewords are all generated internally — the point of this operator
        is the CHANNEL, not the caller's messages. Returns the full
        lattice.fading_trial record:
          'gains'      — the realized fading
          'selected'   — decode of the computation-rate-maximizing combination
                         (value, true, exact, coefficients, power_factor, rate)
          'plain'      — the transmit-for-plain-sum baseline through the same channel
          'deep_fades' — count of |h_i| < 0.2
        """
        from . import lattice as _lattice
        if inputs is None:
            rng = np.random.default_rng()
        elif isinstance(inputs, (int, np.integer)):
            rng = np.random.default_rng(int(inputs))
        else:
            rng = inputs
        return _lattice.fading_trial(
            num_nodes=op.num_nodes, L=op.modulus, n=op.dimension,
            snr_db=op.snr_db, rng=rng, selection=op.selection,
        )

    # ─── OTA aggregation (Lineage 2, Tier 4) ──────────────────────────────

    def _apply_ota_aggregation(self, op: OTAAggregationOperator, inputs) -> dict:
        """
        Aggregate K gradient vectors over the air. inputs is a list of K
        equal-length 1-D arrays. Returns the ota_fl.ota_aggregate record:
        'estimate' (the receiver's average), 'true_average', 'mse', 'nmse',
        'participants', 'max_tx_power', 'channel_uses'.
        """
        from . import ota_fl as _ota
        return _ota.ota_aggregate(
            inputs, snr_db=op.snr_db, rng=np.random.default_rng(op.seed),
            truncation=op.truncation, clip=op.clip, max_phase=op.max_phase,
            max_timing=op.max_timing, receiver=op.receiver,
        )

    # ─── Convolution (Lineage 1) ─────────────────────────────────────────

    def _apply_convolution(self, op: ConvolutionOperator, inputs) -> np.ndarray:
        x = np.asarray(inputs, dtype=np.complex128)
        h = op.impulse_response
        # The wave-domain operation: the medium convolves x with h.
        # In simulation this is numpy convolve; in SDR backend the FIR filter
        # chain does this at RF (see docs/hello-world-convolution.md).
        return np.convolve(x, h, mode='same')

    # ─── Reservoir (Lineage 4) ────────────────────────────────────────────

    def _apply_reservoir(self, op: ReservoirOperator, inputs) -> np.ndarray:
        """
        Approximate software model of spin-torque reservoir computing.

        Generates a random reservoir matrix with the requested spectral
        radius, applies the input, and returns the reservoir state. This is
        an APPROXIMATE model — real spin-torque oscillators have richer
        dynamics (magnetization precession) that this linear-nonlinear
        approximation doesn't capture.
        """
        x = np.atleast_1d(np.asarray(inputs, dtype=np.float64))
        # Build a random reservoir matrix with target spectral radius
        rng = np.random.default_rng(42)  # deterministic for reproducibility
        N = op.reservoir_size
        W = rng.standard_normal((N, N))
        eigs = np.linalg.eigvals(W)
        W = W * (op.spectral_radius / np.max(np.abs(eigs)))
        # Input weights: one scalar input per time step
        W_in = rng.standard_normal(N) * 0.1
        if op.nonlinearity == "tanh":
            f = np.tanh
        else:
            f = lambda s: 1.0 / (1.0 + np.exp(-s))
        # Echo-state recurrence driven by the input sequence:
        #   state_t = f(W @ state_{t-1} + W_in * u_t)
        # spectral_radius sets the fading memory — how long past inputs echo.
        state = np.zeros(N)
        for u in x:
            state = f(W @ state + W_in * u)
        return state

    # ─── Inversion (Lineage 1, closed-loop) ───────────────────────────────

    def _apply_inversion_openloop(self, op: InversionOperator, x) -> np.ndarray:
        """Open-loop multiply y = A·x (the operation, not the inversion)."""
        x = np.asarray(x, dtype=np.float64)
        return op.matrix @ x

    def _solve_inversion(self, op: InversionOperator, b: np.ndarray) -> dict:
        """
        Closed-loop Richardson iteration: x_{k+1} = x_k + alpha*(b - A*x_k).
        The wave domain applies A (convolution in RF); the feedback provides
        the update. Converges to x = A⁻¹·b.
        """
        b = np.asarray(b, dtype=np.float64)
        x = np.zeros_like(b)
        history = []
        alpha = op.step_size

        for k in range(op.max_iters):
            y = op.matrix @ x        # wave domain applies A
            err = b - y               # feedback error
            x = x + alpha * err        # feedback update
            err_norm = float(np.linalg.norm(err))
            history.append(err_norm)
            if err_norm < 1e-6:
                break

        final_err = float(np.linalg.norm(op.matrix @ x - b))
        return {
            'x': x,
            'history': history,
            'converged': final_err < 1e-4,
            'iterations': len(history),
            'final_error': final_err,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience constructors — the named operators from the Tier 2 walkthrough
# ─────────────────────────────────────────────────────────────────────────────

def boxcar(N: int = 64) -> ConvolutionOperator:
    """Moving average — smoothing operator (low-pass)."""
    return ConvolutionOperator(impulse_response=np.ones(N) / N, notes="boxcar smoothing")


def differencer() -> ConvolutionOperator:
    """First difference — differentiation operator (high-pass / edge detection).
    This is the Silva 2014 operation."""
    return ConvolutionOperator(impulse_response=np.array([1.0, -1.0]), notes="differencer; Silva 2014")


def matched(template: np.ndarray) -> ConvolutionOperator:
    """Matched filter — correlation operator (template matching; radar/sonar basis)."""
    template = np.asarray(template, dtype=np.complex128)
    return ConvolutionOperator(impulse_response=np.conj(template[::-1]), notes="matched filter")


def hilbert(N: int = 64) -> ConvolutionOperator:
    """Hilbert transform — analytic-signal operator (phase extraction; SSB basis)."""
    # Ideal discrete Hilbert transformer: h[m] = 2/(pi*m) for odd m, 0 for
    # even m (including the center, m = 0). Filling the even taps too gives
    # a filter whose passband gain swings between ~0.25 and ~1.75.
    m = np.arange(N) - N // 2
    odd = (m % 2) != 0
    h = np.zeros(N)
    h[odd] = 2.0 / (np.pi * m[odd])
    h = h * np.hamming(N)
    return ConvolutionOperator(impulse_response=h, notes="Hilbert transform")


__all__ = [
    'Operator', 'AirCompOperator', 'LatticeAirCompOperator',
    'FadingAirCompOperator', 'OTAAggregationOperator',
    'ConvolutionOperator', 'InversionOperator', 'ReservoirOperator',
    'WaveComputeKernel',
    'boxcar', 'differencer', 'matched', 'hilbert',
]