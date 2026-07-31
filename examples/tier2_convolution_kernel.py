"""
Tier 2 — Wave-domain convolution, using the rf-compute kernel.

Same experiment as docs/hello-world-convolution.md, rewritten with the
kernel. Demonstrates all four named operators (boxcar, differencer, matched,
hilbert) in one script, so you can see one kernel, one apply() call, four
different mathematical operations.

Run:  python examples/tier2_convolution_kernel.py
"""
import numpy as np
from rf_compute import WaveComputeKernel, boxcar, differencer, matched, hilbert

kernel = WaveComputeKernel(backend="sim")

# Input signal: a short pulse followed by a step
x = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             dtype=np.complex128)

print("Wave-domain convolution with four named operators")
print("=" * 60)
print(f"Input x: {np.real(x).astype(int)}")
print()

for op, name in [(boxcar(8), "boxcar (smoothing)"),
                 (differencer(), "differencer (differentiation — Silva 2014)"),
                 (matched(np.array([1, 1, 1, 1])), "matched (correlation)"),
                 (hilbert(16), "Hilbert (phase)"),
                ]:
    y = kernel.apply(op, inputs=x)
    print(f"{name}:")
    print(f"  h[n] = {np.real(op.impulse_response).round(3)}")
    print(f"  y    = {np.real(y[:16]).round(3)}")
    print()

print("Provenance: Silva et al., Science (2014)")
print("Walkthrough: docs/hello-world-convolution.md")