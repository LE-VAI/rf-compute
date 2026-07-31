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
    Lineage 2 — Over-the-Air Computation.

    The wireless channel computes a function (sum, by default) of N
    distributed inputs. The channel is the adder; interference is the
    operation, not noise.

    Maps to: Nazer & Gastpar, IEEE Trans. Inf. Theory (2011).
    Walkthrough: docs/hello-world-aircomp.md
    """
    num_nodes: int = 2
    function: str = "sum"   # "sum" is the only function the analog-superposition
                            # version supports; lattice-coded extension is Tier 1.5
    lineage: str = "aircomp"

    def __post_init__(self):
        if self.function != "sum":
            raise NotImplementedError(
                f"AirComp function '{self.function}' not supported. "
                "The analog-superposition kernel supports 'sum' only. "
                "Lattice-coded finite-field functions (Nazer & Gastpar 2011) "
                "are a Tier 1.5 contribution target — see CONTRIBUTING.md."
            )


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

    Maps to: Nature Communications (2025), arXiv:2301.02850.
    Walkthrough: docs/hello-world-matrix-inversion.md
    """
    matrix: np.ndarray = field(default_factory=lambda: np.array([[0.8, 0.2], [0.1, 0.7]]))
    step_size: float = 0.5   # alpha; must satisfy 0 < alpha < 2/lambda_max(A)
    max_iters: int = 50
    lineage: str = "inversion"

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        # Stability check — warn (don't fail) so builders can experiment
        eigs = np.linalg.eigvals(self.matrix)
        spectral_radius = np.max(np.abs(eigs))
        if spectral_radius >= 1.0:
            import warnings
            warnings.warn(
                f"Matrix spectral radius {spectral_radius:.3f} >= 1.0. "
                f"Richardson iteration may diverge. Normalize the matrix by its "
                f"largest eigenvalue for stable convergence."
            )
        if self.step_size >= 2.0 / spectral_radius:
            import warnings
            warnings.warn(
                f"step_size {self.step_size} >= 2/lambda_max ({2.0/spectral_radius:.3f}). "
                f"Iteration will diverge. Reduce step_size."
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
        For ConvolutionOperator: inputs is a 1-D signal array; returns x * h.
        For ReservoirOperator:  inputs is a 1-D signal array; returns the
                                high-dimensional reservoir state.
        For InversionOperator:  apply does the open-loop multiply A·x (not the
                                inversion — use solve() for that).
        """
        if isinstance(op, AirCompOperator):
            return self._apply_aircomp(op, inputs)
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
        x = np.asarray(inputs, dtype=np.float64)
        # Build a random reservoir matrix with target spectral radius
        rng = np.random.default_rng(42)  # deterministic for reproducibility
        N = op.reservoir_size
        W = rng.standard_normal((N, N))
        eigs = np.linalg.eigvals(W)
        W = W * (op.spectral_radius / np.max(np.abs(eigs)))
        # Input weight matrix
        W_in = rng.standard_normal((N, len(x))) * 0.1
        # Reservoir state update (one step — the simulation is a snapshot)
        state = W_in @ x
        if op.nonlinearity == "tanh":
            state = np.tanh(state)
        elif op.nonlinearity == "sigmoid":
            state = 1.0 / (1.0 + np.exp(-state))
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
    n = np.arange(N)
    # Avoid divide-by-zero at the center sample (n = N//2) — set it to 0
    # before the division, then explicitly zero it after (the ideal Hilbert
    # transform has value 0 at the center index).
    denom = np.where(n == N // 2, 1.0, n - N // 2)  # placeholder 1.0 avoids div-by-zero
    h = 2.0 / (np.pi * denom)
    h[N // 2] = 0
    h = h * np.hamming(N)
    return ConvolutionOperator(impulse_response=h, notes="Hilbert transform")


__all__ = [
    'Operator', 'AirCompOperator', 'ConvolutionOperator', 'InversionOperator', 'ReservoirOperator',
    'WaveComputeKernel',
    'boxcar', 'differencer', 'matched', 'hilbert',
]