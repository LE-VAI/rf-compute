"""
Tier 3 — Matrix inversion via feedback, using the rf-compute kernel.

Same experiment as docs/hello-world-matrix-inversion.md (Mode A: simulation),
rewritten with the kernel. The kernel's solve() method runs the Richardson
iteration; you get the solution, the convergence history, and the final
error in one call.

Run:  python examples/tier3_inversion_kernel.py
"""
import numpy as np
from rf_compute import InversionOperator, WaveComputeKernel

kernel = WaveComputeKernel(backend="sim")

# The operator: a stable 2x2 matrix (spectral radius < 1)
A = np.array([[0.8, 0.2],
              [0.1, 0.7]])
op = InversionOperator(matrix=A, step_size=0.5, max_iters=50)

# The target: solve A·x = b for x
x_true = np.array([1.5, -0.5])
b = A @ x_true
print(f"Target: solve A·x = b, where b = {b}")
print(f"True solution: x = {x_true}")
print()

# The wave-domain solve: feedback loop runs Richardson iteration.
# The wave domain applies A (convolution in RF); the feedback provides the
# update x_{k+1} = x_k + alpha*(b - A*x_k). Converges to x = A⁻¹·b.
result = kernel.solve(op, target=b)

print(f"Converged:    {result['converged']} (after {result['iterations']} iterations)")
print(f"Solution:     x = {result['x'].round(4)}")
print(f"True:         x = {x_true}")
print(f"Final error:  ||A·x - b|| = {result['final_error']:.6f}")
print()
print("Provenance: Nature Communications (2025), arXiv:2301.02850")
print("Walkthrough: docs/hello-world-matrix-inversion.md")