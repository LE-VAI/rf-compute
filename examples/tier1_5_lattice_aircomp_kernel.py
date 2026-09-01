"""
Tier 1.5 — Lattice-coded AirComp, using the rf-compute kernel.

The Tier 1 example showed ANALOG superposition: the receiver reads a
real-valued sum estimate from the waveform. This is the upgrade the theory
promised: with nested lattice codes the same superposition yields an EXACT
integer sum mod L — noise-resilient, one channel use, and the receiver
decodes no individual message.

The kernel call is one line; the operator carries the lineage.

Run:  python examples/tier1_5_lattice_aircomp_kernel.py
"""
import numpy as np

from rf_compute import LatticeAirCompOperator, WaveComputeKernel
from rf_compute.lattice import monte_carlo

# Free, no hardware. The SDR backend transmits the same codewords over RF;
# see docs/hello-world-lattice-aircomp.md for sync and dither-seed sharing.
kernel = WaveComputeKernel(backend="sim")

print("=" * 62)
print("Tier 1.5 — nested-lattice AirComp (Nazer & Gastpar 2007 / 2011)")
print("=" * 62)

# ── 1. One round: the channel computes the exact sum ──────────────────────
# Two nodes hold integers mod 16. The channel superposes their dithered
# lattice codewords; the receiver replays the shared dithers and reads
# w1 + w2 mod 16 — exactly.
op = LatticeAirCompOperator(num_nodes=2, modulus=16, dimension=32)
w1, w2 = 11, 9
out = kernel.apply(op, inputs=[w1, w2])

print()
print("1. ONE CHANNEL USE, EXACT SUM")
print(f"   messages:  w = [{w1}, {w2}]  (mod {op.modulus})")
print(f"   decoded:   {out['value']}   true: {out['true_value']}   "
      f"exact: {out['exact']}")
print(f"   the receiver recovered the SUM — and cannot recover w1 or w2:")
print(f"   from the channel output alone, neither message exists.")

# ── 2. The compute-and-forward equation: arbitrary integer coefficients ───
# With coefficients [a1..aN] the channel computes sum(a_i * w_i) mod L.
# Integer coefficients ride the channel as gains — the equation is free.
op2 = LatticeAirCompOperator(num_nodes=2, modulus=16, dimension=32,
                             coefficients=[1, 2])
out2 = kernel.apply(op2, inputs=[5, 9])

print()
print("2. THE COMPUTE-AND-FORWARD EQUATION")
print(f"   w = [5, 9], a = [1, 2]  ->  1*5 + 2*9 = 23 = 7 (mod 16)")
print(f"   decoded:   {out2['value']}   true: {out2['true_value']}   "
      f"exact: {out2['exact']}")

# ── 3. Reliability: error decays exponentially in the lattice dimension ──
print()
print("3. ERROR vs LATTICE DIMENSION n   (2 nodes, L=16, 5 dB, 2000 trials)")
print("   each coordinate is an independent vote on the same sum;")
print("   the circular-mean decoder banks the sqrt(n) gain.")
print(f"   {'n':>4}  {'error':>8}  {'channel uses':>12}")
rng_seed = 20
for n in [4, 8, 16, 32, 64]:
    mc = monte_carlo(num_nodes=2, L=16, n=n, snr_db=5.0, trials=2000,
                     seed=rng_seed + n)
    print(f"   {n:>4}  {mc['lattice']['error_rate']:>8.4f}  "
          f"{mc['lattice']['channel_uses_per_trial']:>12}")

# ── 4. The honest scoreline: all three schemes at matched power ────────────
print()
print("4. THE HONEST SCORELINE   (2 nodes, L=16, n=32, 5 dB, 4000 trials)")
print("   lattice: exact sum, ONE channel use")
print("   analog:  1 use, but no mod-L arithmetic + power-shaping gap")
print("   tdma:    N uses, exact per slot, N-way error union")
print(f"   {'scheme':>8}  {'error':>8}  {'channel uses':>12}")
mc = monte_carlo(num_nodes=2, L=16, n=32, snr_db=5.0, trials=4000, seed=42)
for scheme in ['lattice', 'analog', 'tdma']:
    print(f"   {scheme:>8}  {mc[scheme]['error_rate']:>8.4f}  "
          f"{mc[scheme]['channel_uses_per_trial']:>12}")

print()
print("Provenance: Nazer & Gastpar, IEEE Trans. Inf. Theory 53(10) 3498 (2007),")
print("             57(10) 6463 (2011), DOI 10.1109/TIT.2011.2165816")
print("Walkthrough: docs/hello-world-lattice-aircomp.md")