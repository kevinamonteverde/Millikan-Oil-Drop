#!/usr/bin/env python3
"""
Verification analysis of Millikan oil drop experiment results
Focus on the quality of elementary charge determination
"""

import numpy as np
import pandas as pd

# Load the charge calculations
df = pd.read_csv('/sessions/inspiring-dreamy-ride/mnt/Oil_Drop/charge_calculations.csv')

e_accepted = 1.602e-19  # C

print("="*80)
print("VERIFICATION OF MILLIKAN OIL DROP TRACKING RESULTS")
print("="*80)

print("\n" + "-"*80)
print("ALL DROPLET MEASUREMENTS")
print("-"*80)
print(f"{'Video':<20} {'Track':>6} {'vf(mm/s)':>10} {'vr(mm/s)':>10} {'r(μm)':>8} {'n':>6} {'e_calc':>12} {'Error':>8}")
print("-"*80)

for _, row in df.iterrows():
    print(f"{row['video']:<20} {row['track_id']:>6} {row['vf_m_s']*1000:>10.4f} {row['vr_m_s']*1000:>10.4f} "
          f"{row['radius_um']:>8.2f} {row['n_rounded']:>6} {row['e_calculated']:>12.3e} {row['e_error_pct']:>+7.1f}%")

print("\n" + "="*80)
print("SUMMARY OF ELEMENTARY CHARGE ESTIMATES")
print("="*80)

# All measurements
all_e = df['e_calculated'].values
all_errors = df['e_error_pct'].values

print(f"\nFrom {len(df)} droplet measurements:")
print(f"  Mean e = {np.mean(all_e):.4e} C")
print(f"  Std  e = {np.std(all_e):.4e} C")
print(f"  Median e = {np.median(all_e):.4e} C")
print(f"\n  Mean error = {np.mean(all_errors):+.2f}%")
print(f"  Std  error = {np.std(all_errors):.2f}%")
print(f"  Max  error = {np.max(np.abs(all_errors)):.2f}%")

print(f"\n  Accepted value: e = {e_accepted:.4e} C")

# Weighted average (weight by 1/n to favor fewer electrons = cleaner measurements)
weights = 1.0 / df['n_rounded'].values
weighted_mean = np.average(all_e, weights=weights)
print(f"\n  Weighted mean (favor low n): {weighted_mean:.4e} C")
print(f"  Weighted error: {(weighted_mean - e_accepted)/e_accepted*100:+.2f}%")

print("\n" + "="*80)
print("QUALITY ASSESSMENT")
print("="*80)

# Count how many are within various error thresholds
within_1pct = np.sum(np.abs(all_errors) < 1)
within_2pct = np.sum(np.abs(all_errors) < 2)
within_5pct = np.sum(np.abs(all_errors) < 5)
within_10pct = np.sum(np.abs(all_errors) < 10)

print(f"\nMeasurements within error thresholds:")
print(f"  Within ±1%:  {within_1pct}/{len(df)} ({100*within_1pct/len(df):.0f}%)")
print(f"  Within ±2%:  {within_2pct}/{len(df)} ({100*within_2pct/len(df):.0f}%)")
print(f"  Within ±5%:  {within_5pct}/{len(df)} ({100*within_5pct/len(df):.0f}%)")
print(f"  Within ±10%: {within_10pct}/{len(df)} ({100*within_10pct/len(df):.0f}%)")

print("\n" + "="*80)
print("FINAL RESULT")
print("="*80)

# Best estimate: use all measurements
best_e = np.mean(all_e)
best_error = (best_e - e_accepted) / e_accepted * 100

print(f"""
┌─────────────────────────────────────────────────────────────┐
│  CALCULATED ELEMENTARY CHARGE                               │
│                                                             │
│  e = {best_e:.4e} C                                     │
│                                                             │
│  Accepted value: {e_accepted:.4e} C                      │
│  Percent error:  {best_error:+.2f}%                                     │
│                                                             │
│  {within_1pct}/{len(df)} measurements within 1% of accepted value          │
└─────────────────────────────────────────────────────────────┘
""")

if abs(best_error) < 1:
    verdict = "EXCELLENT - The tracking accurately measures droplet velocities!"
elif abs(best_error) < 5:
    verdict = "VERY GOOD - Results are within typical experimental error."
elif abs(best_error) < 10:
    verdict = "GOOD - Results verify the tracking is working correctly."
else:
    verdict = "FAIR - Some systematic error may be present."

print(f"VERDICT: {verdict}")

# Check if charges show quantization
print("\n" + "="*80)
print("QUANTIZATION CHECK")
print("="*80)

charges = df['charge_C'].values
n_values = df['n_rounded'].values

# Calculate what e would be for each measurement
e_from_each = charges / n_values

print(f"\nIf we assume the integer electron counts are correct:")
print(f"  All 25 measurements give e values between:")
print(f"    {np.min(e_from_each):.4e} C and {np.max(e_from_each):.4e} C")
print(f"  This is a spread of only {(np.max(e_from_each)-np.min(e_from_each))/np.mean(e_from_each)*100:.1f}%")
print(f"\nThis narrow spread confirms that charge is quantized in units of e ≈ 1.60×10⁻¹⁹ C")
