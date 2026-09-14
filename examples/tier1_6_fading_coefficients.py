"""
Tier 1.6 — Fading-channel AirComp: the coefficients must be EARNED.

Tier 1.5 set the channel gains to integers (h_i = a_i). That made the
compute-and-forward equation exact — and hid the actual problem: on a real
channel the gains fade, and the receiver must CHOOSE which integer
combination to decode.

This example runs the choice. The machinery is in rf_compute.coefficients:

    computation_rate(h, a, snr_db)   — the Nazer/Gastpar rate for one a
    mmse_alpha(h, a, snr_db)         — the optimal receiver scaling
    select_coefficients(h, snr_db)   — exhaustive / LLL / nearest-integer

The honest finding, visible in the numbers below: on a fading channel the
plain sum (a = 1 vector) has computation rate ZERO most of the time — its
alignment is wrong for the realized channel. Coefficient selection finds a
decodable equation instead, and it can DROP a faded node (a_i = 0) rather
than burn power inverting a deep fade.

Run:  python examples/tier1_6_fading_coefficients.py
"""
import numpy as np

from rf_compute.coefficients import (
    computation_rate, mmse_alpha, norm_bound, select_coefficients, fading_gains,
)
from rf_compute.lattice import fading_scoreline
from rf_compute import FadingAirCompOperator, WaveComputeKernel

kernel = WaveComputeKernel(backend="sim")

print("=" * 68)
print("Tier 1.6 — fading-channel coefficients (Nazer & Gastpar 2011)")
print("=" * 68)

# ── 1. One fading realization: the selection problem in the open ──────────
rng = np.random.default_rng(3)
h = fading_gains(3, rng)
print()
print("1. ONE FADING CHANNEL  (h_i ~ N(0,1), 3 nodes, 5 dB)")
print(f"   gains:  {', '.join(f'{x:+.3f}' for x in h)}")
print(f"   norm bound ||a|| <= sqrt(1 + SNR||h||^2) = {norm_bound(h, 5.0):.2f}")
print(f"   {'method':>11}  {'a':>10}  {'rate (bits/use)':>15}")
for method in ["exhaustive", "lll", "rounded"]:
    a, r = select_coefficients(h, 5.0, method=method)
    print(f"   {method:>11}  {str(list(a)):>10}  {r:>15.4f}")
a_plain = np.ones(3, dtype=int)
print(f"   {'plain sum':>11}  {str(list(a_plain)):>10}  "
      f"{computation_rate(h, a_plain, 5.0):>15.4f}   <- nothing selected")

# ── 2. Why alpha matters (and why alpha = 1 is not optimal) ───────────────
print()
print("2. THE RECEIVER SCALING alpha  (why the theory optimizes it)")
print("   Tier 1.5 fixes alpha = 1 and relies on h = a making the self-noise")
print("   vanish identically. On a real channel no alpha aligns perfectly:")
h1 = np.array([1.0, 2.0])
a12 = np.array([1, 2])
al = mmse_alpha(h1, a12, 5.0)
print(f"   h = [1, 2], a = [1, 2]:  alpha* = {al:.4f}  (< 1)")
print(f"     rate(alpha*)  = {computation_rate(h1, a12, 5.0):.4f}")
print(f"     rate(alpha=1) = {computation_rate(h1, a12, 5.0, alpha=1.0):.4f}")
print(f"   selection instead drops the weaker node: a = [0, 1]")
print(f"     rate          = {computation_rate(h1, np.array([0, 1]), 5.0):.4f}")

# ── 3. Over many channels: selection vs the plain sum ────────────────────
print()
print("3. OVER 800 INDEPENDENT FADING CHANNELS  (3 nodes, 5 dB)")
fs = fading_scoreline(num_nodes=3, L=16, n=32, snr_db=5.0, trials=800,
                      seed=11, selection="exhaustive")
print(f"   {'strategy':>10}  {'mean rate':>10}  {'power med':>10}  {'power mean':>11}")
print(f"   {'selected':>10}  {fs['selected']['mean_rate_bits']:>10.4f}  "
      f"{fs['selected']['median_power_factor']:>10.2f}  "
      f"{fs['selected']['mean_power_factor']:>11.2f}")
print(f"   {'plain sum':>10}  {fs['plain']['mean_rate_bits']:>10.4f}  "
      f"{fs['plain']['median_power_factor']:>10.2f}  "
      f"{fs['plain']['mean_power_factor']:>11.2f}")
print()
print(f"   plain-sum rate is ZERO on {100*fs['plain_rate_zero_fraction']:.1f}% of channels")
print(f"   selection returned the plain sum on "
      f"{100*fs['plain_was_optimal_fraction']:.1f}% of channels")
print("   power factor = sum (a_i/h_i)^2 / N  (1.0 = no fading penalty).")
print("   The MEAN is dominated by rare deep fades; the MEDIAN is the")
print("   typical cost. Selection drops faded nodes (a_i = 0); the plain")
print("   sum must invert every fade, including the deep ones.")

# ── 4. Selection vs no selection, through the kernel, on ONE channel ─────
print()
print("4. ONE CHANNEL, BOTH STRATEGIES  (kernel, seeded)")
op = FadingAirCompOperator(num_nodes=3, modulus=16, dimension=32, snr_db=5.0,
                           selection="exhaustive")
out = kernel.apply(op, 3)
print(f"   gains:    {', '.join(f'{x:+.2f}' for x in out['gains'])}")
print(f"   selected: a = {out['selected']['coefficients']}  "
      f"rate = {out['selected']['rate']:.4f}  "
      f"power = {out['selected']['power_factor']:.2f}x  "
      f"exact = {out['selected']['exact']}")
print(f"   plain:    a = {out['plain']['coefficients']}  "
      f"rate = {out['plain']['rate']:.4f}  "
      f"power = {out['plain']['power_factor']:.2f}x  "
      f"exact = {out['plain']['exact']}")

print()
print("Read the rate and power columns, not the error columns. In this")
print("inverted toy BOTH strategies decode at the same ~30% threshold —")
print("the pre-code re-aligns the integer gains, so fading does not add")
print("error here. What selection changes is:")
print("  (1) DECODABILITY — the plain sum's rate is 0 in the real system")
print("      on most channels; no decoder can recover it there;")
print("  (2) POWER — the plain sum must invert every fade, including the")
print("      deep ones; selection DROPS a faded node (a_i = 0) instead.")
print()
print("Provenance: Nazer & Gastpar, IEEE TIT 57(10) 6463 (2011);")
print("            Sahraei & Gastpar, Allerton 2014 (arXiv:1410.3656);")
print("            Liu & Ling, IEEE TWC 15(12) 8039 (2016);")
print("            Huang & Burr, arXiv:1704.05007 (MMSE alpha, rate formula)")
print("Walkthrough: docs/hello-world-fading-coefficients.md")
