#!/usr/bin/env python3
"""
Calculate the elementary charge from Millikan oil drop experiment data
Using the formula from the PASCO AP-8210A manual (page 11)

Formula:
q = (4/3) * π * ρ * g * [√((b/2p)² + 9ηvf/(2gρ)) - b/(2p)]³ × (vf + vr)/(E * vf)

Where E = V/d
"""

import numpy as np
import pandas as pd

# Physical constants
g = 9.81  # m/s² - acceleration due to gravity
e_accepted = 1.602e-19  # C - accepted value of elementary charge

# Apparatus parameters (from manual - typical values for PASCO AP-8210A)
d = 7.6e-3  # m - plate separation (~7.6 mm per manual)

# Oil properties (Squibb #5597 Mineral Oil)
rho = 886  # kg/m³ - density of oil

# Air properties at ~22°C (typical room temperature)
# From Appendix A of the manual: η ≈ 1.82 × 10⁻⁵ N·s/m²
eta = 1.82e-5  # N·s/m² - viscosity of air

# Stokes' Law correction constant
b = 8.20e-3  # Pa·m (from manual page 11)

# Atmospheric pressure (standard)
p = 101325  # Pa

# Typical plate voltage
V = 500  # V (typical operating voltage)

# Electric field
E = V / d  # V/m

print("="*80)
print("MILLIKAN OIL DROP EXPERIMENT - ELEMENTARY CHARGE CALCULATION")
print("="*80)
print("\nApparatus Parameters:")
print(f"  Plate separation (d): {d*1000:.2f} mm")
print(f"  Plate voltage (V): {V} V")
print(f"  Electric field (E): {E:.1f} V/m")
print(f"  Oil density (ρ): {rho} kg/m³")
print(f"  Air viscosity (η): {eta:.2e} N·s/m²")
print(f"  Pressure (p): {p} Pa")
print(f"  Correction constant (b): {b:.2e} Pa·m")
print(f"\nAccepted value of e: {e_accepted:.3e} C")


def calculate_drop_radius(vf, eta, rho, g, b, p):
    """
    Calculate the corrected radius of the oil drop using Stokes' Law with correction

    a = √((b/2p)² + 9ηvf/(2gρ)) - b/(2p)
    """
    term1 = (b / (2 * p))**2
    term2 = (9 * eta * vf) / (2 * g * rho)
    a = np.sqrt(term1 + term2) - b / (2 * p)
    return a


def calculate_charge(vf, vr, E, rho, g, eta, b, p):
    """
    Calculate the charge on a droplet using the Millikan formula

    q = (4/3) * π * ρ * g * a³ × (vf + vr)/(E * vf)

    where a is the corrected radius
    """
    # Calculate corrected radius
    a = calculate_drop_radius(vf, eta, rho, g, b, p)

    # Calculate charge
    q = (4/3) * np.pi * rho * g * (a**3) * (vf + vr) / (E * vf)

    return q, a


def analyze_charges(charges):
    """
    Analyze a list of charges to find the elementary charge
    by looking for the GCD (greatest common divisor pattern)
    """
    charges = np.array(charges)
    charges = charges[charges > 0]  # Remove any negative or zero charges

    # Sort charges
    charges_sorted = np.sort(charges)

    # Find differences between consecutive charges
    # These differences should be multiples of e

    # Method 1: Look for smallest charge
    min_charge = np.min(charges)

    # Method 2: Look at ratios to find common factor
    # Divide all charges by smallest
    ratios = charges / min_charge

    # Round to nearest integer to estimate number of electrons
    n_electrons = np.round(ratios)

    # Recalculate e from each measurement
    e_estimates = charges / n_electrons

    return {
        'min_charge': min_charge,
        'ratios': ratios,
        'n_electrons': n_electrons,
        'e_estimates': e_estimates,
        'e_mean': np.mean(e_estimates),
        'e_std': np.std(e_estimates)
    }


# Load the velocity data
df = pd.read_csv('/sessions/inspiring-dreamy-ride/mnt/Oil_Drop/millikan_summary.csv')

print("\n" + "="*80)
print("CHARGE CALCULATIONS FOR EACH DROPLET")
print("="*80)

results = []

for idx, row in df.iterrows():
    vf = row['fall_velocity_mm_s'] * 1e-3  # Convert mm/s to m/s
    vr = row['rise_velocity_mm_s'] * 1e-3  # Convert mm/s to m/s

    if pd.isna(vf) or pd.isna(vr) or vf <= 0 or vr <= 0:
        continue

    q, a = calculate_charge(vf, vr, E, rho, g, eta, b, p)

    # Number of elementary charges (estimated)
    n = q / e_accepted
    n_rounded = round(n)

    # Recalculate e based on integer n
    if n_rounded > 0:
        e_calc = q / n_rounded
    else:
        e_calc = q

    results.append({
        'video': row['video'],
        'track_id': row['track_id'],
        'vf_m_s': vf,
        'vr_m_s': vr,
        'radius_um': a * 1e6,  # Convert to micrometers
        'charge_C': q,
        'n_estimated': n,
        'n_rounded': n_rounded,
        'e_calculated': e_calc,
        'e_error_pct': (e_calc - e_accepted) / e_accepted * 100
    })

    print(f"\n{row['video']} - Track {row['track_id']}:")
    print(f"  vf = {vf*1000:.4f} mm/s, vr = {vr*1000:.4f} mm/s")
    print(f"  Droplet radius: {a*1e6:.3f} μm")
    print(f"  Charge q = {q:.3e} C")
    print(f"  n ≈ {n:.2f} → {n_rounded} electrons")
    if n_rounded > 0:
        print(f"  e = q/n = {e_calc:.3e} C (error: {(e_calc - e_accepted)/e_accepted*100:+.1f}%)")

results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("STATISTICAL ANALYSIS")
print("="*80)

# Filter for reasonable results (1-10 electrons, reasonable error)
valid = results_df[
    (results_df['n_rounded'] >= 1) &
    (results_df['n_rounded'] <= 20) &
    (results_df['e_error_pct'].abs() < 100)  # Within 100% error
]

print(f"\nTotal droplets analyzed: {len(results_df)}")
print(f"Valid measurements (1-20 electrons, <100% error): {len(valid)}")

if len(valid) > 0:
    print(f"\nDistribution of electron counts:")
    print(valid['n_rounded'].value_counts().sort_index())

    print(f"\nElementary charge estimates:")
    print(f"  Mean: {valid['e_calculated'].mean():.3e} C")
    print(f"  Std:  {valid['e_calculated'].std():.3e} C")
    print(f"  Min:  {valid['e_calculated'].min():.3e} C")
    print(f"  Max:  {valid['e_calculated'].max():.3e} C")

    mean_e = valid['e_calculated'].mean()
    error = (mean_e - e_accepted) / e_accepted * 100
    print(f"\n  Accepted value: {e_accepted:.3e} C")
    print(f"  Our mean value: {mean_e:.3e} C")
    print(f"  Percent error:  {error:+.1f}%")

    # Try to find GCD pattern
    print("\n" + "="*80)
    print("VERIFICATION: LOOKING FOR QUANTIZATION")
    print("="*80)

    charges = valid['charge_C'].values
    analysis = analyze_charges(charges)

    print(f"\nSmallest measured charge: {analysis['min_charge']:.3e} C")
    print(f"Ratio to accepted e: {analysis['min_charge']/e_accepted:.2f}")

    print(f"\nCharge ratios (normalized to smallest):")
    for i, (q, ratio, n) in enumerate(zip(charges, analysis['ratios'], analysis['n_electrons'])):
        print(f"  q = {q:.3e} C, ratio = {ratio:.2f}, n = {int(n)}")

    print(f"\nEstimated elementary charge from GCD analysis:")
    print(f"  Mean e = {analysis['e_mean']:.3e} C")
    print(f"  Std e  = {analysis['e_std']:.3e} C")
    print(f"  Error  = {(analysis['e_mean'] - e_accepted)/e_accepted*100:+.1f}%")

# Save results
results_df.to_csv('/sessions/inspiring-dreamy-ride/mnt/Oil_Drop/charge_calculations.csv', index=False)
print(f"\nResults saved to: charge_calculations.csv")

# Final assessment
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if len(valid) > 0:
    mean_e = valid['e_calculated'].mean()
    error = abs((mean_e - e_accepted) / e_accepted * 100)

    if error < 10:
        print("\n✓ EXCELLENT: Calculated charge is within 10% of accepted value!")
        print("  The tracking data appears to be accurate.")
    elif error < 25:
        print("\n✓ GOOD: Calculated charge is within 25% of accepted value.")
        print("  The tracking data is reasonably accurate.")
    elif error < 50:
        print("\n⚠ FAIR: Calculated charge is within 50% of accepted value.")
        print("  There may be some calibration or tracking issues.")
    else:
        print("\n✗ POOR: Large deviation from accepted value.")
        print("  Possible issues:")
        print("  - Pixel-to-mm calibration may be incorrect")
        print("  - Plate voltage or separation needs verification")
        print("  - Oil density may differ from assumed value")
        print("  - Droplet velocities may have tracking errors")
else:
    print("\nInsufficient valid data for analysis.")
