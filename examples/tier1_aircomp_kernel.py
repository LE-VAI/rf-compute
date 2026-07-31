"""
Tier 1 — AirComp sum, using the rf-compute kernel.

This is the same experiment as docs/hello-world-aircomp.md, rewritten with
the kernel. Compare the two: the kernel hides the backend choice, the
operator carries the lineage, and apply() is one line.

Run:  python examples/tier1_aircomp_kernel.py
"""
from rf_compute import AirCompOperator, WaveComputeKernel

# Free, no hardware. Switch to backend="sdr" after running the Tier 1
# walkthrough channel calibration.
kernel = WaveComputeKernel(backend="sim")

# The operator carries its lineage explicitly — the channel computes a sum
# of 2 distributed inputs (Nazer & Gastpar 2011).
op = AirCompOperator(num_nodes=2, function="sum")

# The wave-domain operation: the channel computes f(x1, x2).
# In simulation this is numpy sum; in the SDR backend it's real RF superposition.
x1, x2 = 3.0, 5.0
result = kernel.apply(op, inputs=[x1, x2])

print(f"AirComp sum: {x1} + {x2} = {result}")
print(f"(The channel computed the sum without decoding either input.)")
print()
print(f"Provenance: Nazer & Gastpar, IEEE Trans. Inf. Theory (2011)")
print(f"Walkthrough: docs/hello-world-aircomp.md")