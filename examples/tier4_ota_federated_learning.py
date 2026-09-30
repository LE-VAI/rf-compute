"""
Tier 4 — Over-the-air federated learning: the adder goes to work.

Tiers 1-1.6 built the primitive: the channel computes a sum. Here 20
devices train a shared logistic-regression model, and every round their
gradients are summed BY THE CHANNEL — all devices transmit at once.

Then the hardware problem Tier 1's clock cable exists for: misalignment.
Devices arrive with residual phase and timing offsets, and the naive
receiver's sum falls apart. A shared pilot lets the receiver measure the
AGGREGATE channel and equalize it — without ever learning any single
device's offset or gradient.

The machinery is in rf_compute.ota_fl:

    ota_aggregate(grads, ...)        — one over-the-air aggregation
    aggregation_quality(devices, ..) — how well one aggregation recovers the average
    scoreline(...)                   — every scheme, same data, 10 seeds

Run:  python examples/tier4_ota_federated_learning.py   (about 10 seconds)
"""
import numpy as np

from rf_compute import OTAAggregationOperator, WaveComputeKernel
from rf_compute import ota_fl

kernel = WaveComputeKernel(backend="sim")

print("=" * 72)
print("Tier 4 - over-the-air federated learning (Zhu, Wang & Huang 2020;")
print("         misalignment model: Shao, Gunduz & Liew 2022)")
print("=" * 72)

# ── 1. One aggregation through the kernel ─────────────────────────────────
rng = np.random.default_rng(0)
base = rng.normal(size=20)
base *= 0.5 / np.linalg.norm(base)
grads = [base + 0.1 * rng.normal(size=20) for _ in range(20)]

print("\n1. One aggregation of 20 similar gradients, 20 dB SNR")
print(f"   {'channel':<40}{'receiver':<12}{'NMSE':>8}")
for label, ph, tm, rx in (
        ("aligned", 0.0, 0.0, "naive"),
        ("misaligned (phase<=60deg, tau<=0.5)", np.pi / 3, 0.5, "naive"),
        ("misaligned (phase<=60deg, tau<=0.5)", np.pi / 3, 0.5, "equalized")):
    op = OTAAggregationOperator(snr_db=20.0, max_phase=ph, max_timing=tm,
                                receiver=rx, seed=1)
    out = kernel.apply(op, inputs=grads)
    print(f"   {label:<40}{rx:<12}{out['nmse']:>8.4f}")
print(f"   participants {out['participants']}/20 | peak transmit power "
      f"{out['max_tx_power']:.2f} of budget 1.00 | {out['channel_uses']} channel uses")

# ── 2. The equalizer's limit: heterogeneity ───────────────────────────────
print("\n2. Aggregation NMSE vs how different the devices' data are")
print("   (fixed point, 400 channel draws, 20 dB, phase<=60deg, tau<=0.5)")
print(f"   {'gradient spread':>16}{'aligned':>10}{'naive':>10}{'equalized':>11}")
for het in (0.0, 2.0, 4.0):
    devices, _, _ = ota_fl.make_federated_data(heterogeneity=het,
                                               rng=np.random.default_rng(7))
    q = ota_fl.aggregation_quality(devices, snr_db=20.0, max_phase=np.pi / 3,
                                   max_timing=0.5)
    print(f"   {ota_fl.gradient_spread(devices):>16.2f}{q['ota_aligned']:>10.4f}"
          f"{q['ota_misaligned']:>10.3f}{q['ota_equalized']:>11.4f}")

# ── 3. Training: does the aggregation error matter? ───────────────────────
names = {
    'orthogonal': 'orthogonal, error-free (upper bound)',
    'ota_aligned': 'over the air, aligned',
    'ota_misaligned': 'over the air, misaligned, naive',
    'ota_equalized': 'over the air, misaligned, equalized',
}
for title, ph, tm in (("moderate: phase <= 60 deg, timing <= 0.5 symbol", np.pi / 3, 0.5),
                      ("severe: no phase sync, timing <= 1 symbol", np.pi, 1.0)):
    s = ota_fl.scoreline(max_phase=ph, max_timing=tm)
    print(f"\n3. Training, {title}")
    print(f"   20 devices, 60 rounds, 20 dB, gradient spread {s['gradient_spread']:.2f}, 10 seeds")
    print(f"   {'scheme':<38}{'accuracy (min-max)':>20}{'channel uses':>14}")
    for k, label in names.items():
        r = s[k]
        acc = f"{r['accuracy_mean']:.3f} ({r['accuracy_min']:.2f}-{r['accuracy_max']:.2f})"
        print(f"   {label:<38}{acc:>20}{r['channel_uses']:>14}")

print("""
What the numbers say:
  - Over the air, one aggregation costs d+1 samples plus a pilot however
    many devices transmit: 3,180 channel uses for the whole run against
    24,000 for orthogonal slots, at the same accuracy.
  - Moderate misalignment ruins the naive SUM (NMSE about 0.15) but not the
    LEARNING: a consistent shrink acts like a smaller step size.
  - With no phase sync the naive receiver's expected estimate is zero, and
    training becomes a coin flip. The pilot-aided equalizer restores
    upper-bound accuracy - even where its per-round error is large,
    because that error is fresh every round rather than a fixed bias.
  - The equalizer is exact only for what the devices have in COMMON. Its
    error grows with the gradient spread: sync hardware shrinks the
    offsets; nothing at a one-sample receiver removes the heterogeneity.
""")
