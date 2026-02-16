#!/usr/bin/env python3
"""
Recalculate Millikan oil drop charges with corrected experimental parameters
and full error propagation.

Corrected parameters (from user):
  d = 7.5 ± 0.1 mm  (measured with micrometer)
  V = 500 ± 0.1 V   (measured with DMM)
  T ≈ 20°C           (thermistor reading)

Error propagation: numerical perturbation method (perturb each parameter
by ±δ, compute Δq, sum in quadrature).
"""

import numpy as np
import pandas as pd

# =====================================================================
# CORRECTED physical constants and their uncertainties
# =====================================================================
params = {
    'g':   (9.81,       0.001),       # m/s², negligible
    'd':   (7.5e-3,     0.1e-3),      # m, measured with micrometer (0.1 mm resolution)
    'V':   (500.0,      0.1),         # V, measured with DMM
    'rho': (886.0,      5.0),         # kg/m³, PASCO manual notes lot variation
    'eta': (1.813e-5,   0.004e-5),    # N·s/m², at 20°C from Appendix A; ±1°C → ±0.004e-5
    'b':   (8.20e-3,    0.08e-3),     # Pa·m, ~1% assumed (manual gives no uncertainty)
    'p':   (101325.0,   1000.0),      # Pa, ~1% typical barometric variation
}

# Relative uncertainty for CV-extracted velocities
# Sources: pixel calibration (~2-3%), centroid sub-pixel accuracy,
# segment-to-segment scatter. 5% is a conservative estimate.
REL_VEL_UNCERT = 0.05

e_accepted = 1.602e-19  # C

# =====================================================================
# Core calculation functions
# =====================================================================
def calc_radius(vf, eta, rho, g, b, p):
    """Cunningham-corrected Stokes radius."""
    term1 = (b / (2 * p))**2
    term2 = (9 * eta * vf) / (2 * g * rho)
    return np.sqrt(term1 + term2) - b / (2 * p)

def calc_charge(vf, vr, d, V, rho, g, eta, b, p):
    """Full charge calculation. Returns (q, a)."""
    E = V / d
    a = calc_radius(vf, eta, rho, g, b, p)
    q = (4.0/3.0) * np.pi * rho * g * (a**3) * (vf + vr) / (E * vf)
    return q, a

# =====================================================================
# Load existing velocity data (vf, vr are in mm/s in the CSV)
# =====================================================================
df = pd.read_csv('millikan_final_data.csv')

# Central parameter values
g0, eta0, rho0, b0, p0 = params['g'][0], params['eta'][0], params['rho'][0], params['b'][0], params['p'][0]
d0, V0 = params['d'][0], params['V'][0]

print("=" * 70)
print("RECALCULATION WITH CORRECTED PARAMETERS + ERROR PROPAGATION")
print("=" * 70)
print(f"\nCorrected parameters:")
print(f"  d   = {d0*1000:.1f} ± {params['d'][1]*1000:.1f} mm   (was 7.6 mm)")
print(f"  V   = {V0:.1f} ± {params['V'][1]:.1f} V")
print(f"  η   = {eta0:.3e} ± {params['eta'][1]:.1e} N·s/m²  (at 20°C; was 1.82e-5)")
print(f"  ρ   = {rho0:.0f} ± {params['rho'][1]:.0f} kg/m³")
print(f"  b   = {b0:.2e} ± {params['b'][1]:.0e} Pa·m")
print(f"  p   = {p0:.0f} ± {params['p'][1]:.0f} Pa")
print(f"  g   = {g0:.3f} ± {params['g'][1]:.3f} m/s²")
print(f"  δvf/vf = δvr/vr = {REL_VEL_UNCERT*100:.0f}%")

# =====================================================================
# Recalculate central values
# =====================================================================
results = []

for _, row in df.iterrows():
    vf = row['vf_mm_s'] * 1e-3  # mm/s → m/s
    vr = row['vr_mm_s'] * 1e-3

    # Central calculation
    q0, a0 = calc_charge(vf, vr, d0, V0, rho0, g0, eta0, b0, p0)

    # --- Numerical error propagation ---
    # Perturb each parameter by ±δ, compute |Δq|/2 for each
    delta_q_components = {}

    # Helper: perturb one parameter
    def dq_from(name, val_plus, val_minus):
        """Compute half-difference in q when parameter is perturbed ±."""
        if name == 'd':
            qp, _ = calc_charge(vf, vr, val_plus, V0, rho0, g0, eta0, b0, p0)
            qm, _ = calc_charge(vf, vr, val_minus, V0, rho0, g0, eta0, b0, p0)
        elif name == 'V':
            qp, _ = calc_charge(vf, vr, d0, val_plus, rho0, g0, eta0, b0, p0)
            qm, _ = calc_charge(vf, vr, d0, val_minus, rho0, g0, eta0, b0, p0)
        elif name == 'rho':
            qp, _ = calc_charge(vf, vr, d0, V0, val_plus, g0, eta0, b0, p0)
            qm, _ = calc_charge(vf, vr, d0, V0, val_minus, g0, eta0, b0, p0)
        elif name == 'g':
            qp, _ = calc_charge(vf, vr, d0, V0, rho0, val_plus, eta0, b0, p0)
            qm, _ = calc_charge(vf, vr, d0, V0, rho0, val_minus, eta0, b0, p0)
        elif name == 'eta':
            qp, _ = calc_charge(vf, vr, d0, V0, rho0, g0, val_plus, b0, p0)
            qm, _ = calc_charge(vf, vr, d0, V0, rho0, g0, val_minus, b0, p0)
        elif name == 'b':
            qp, _ = calc_charge(vf, vr, d0, V0, rho0, g0, eta0, val_plus, p0)
            qm, _ = calc_charge(vf, vr, d0, V0, rho0, g0, eta0, val_minus, p0)
        elif name == 'p':
            qp, _ = calc_charge(vf, vr, d0, V0, rho0, g0, eta0, b0, val_plus)
            qm, _ = calc_charge(vf, vr, d0, V0, rho0, g0, eta0, b0, val_minus)
        elif name == 'vf':
            qp, _ = calc_charge(val_plus, vr, d0, V0, rho0, g0, eta0, b0, p0)
            qm, _ = calc_charge(val_minus, vr, d0, V0, rho0, g0, eta0, b0, p0)
        elif name == 'vr':
            qp, _ = calc_charge(vf, val_plus, d0, V0, rho0, g0, eta0, b0, p0)
            qm, _ = calc_charge(vf, val_minus, d0, V0, rho0, g0, eta0, b0, p0)
        else:
            return 0.0
        return abs(qp - qm) / 2.0

    # Apparatus parameters
    for name, (val, delta) in params.items():
        delta_q_components[name] = dq_from(name, val + delta, val - delta)

    # Velocity uncertainties
    dvf = vf * REL_VEL_UNCERT
    dvr = vr * REL_VEL_UNCERT
    delta_q_components['vf'] = dq_from('vf', vf + dvf, vf - dvf)
    delta_q_components['vr'] = dq_from('vr', vf, vr)  # placeholder, fix below

    # vr perturbation
    qp_vr, _ = calc_charge(vf, vr + dvr, d0, V0, rho0, g0, eta0, b0, p0)
    qm_vr, _ = calc_charge(vf, vr - dvr, d0, V0, rho0, g0, eta0, b0, p0)
    delta_q_components['vr'] = abs(qp_vr - qm_vr) / 2.0

    # Total uncertainty in q (quadrature sum)
    dq = np.sqrt(sum(c**2 for c in delta_q_components.values()))

    # Radius uncertainty (same numerical approach)
    a_plus_eta, a_minus_eta = calc_radius(vf, eta0 + params['eta'][1], rho0, g0, b0, p0), calc_radius(vf, eta0 - params['eta'][1], rho0, g0, b0, p0)
    a_plus_vf, a_minus_vf = calc_radius(vf + dvf, eta0, rho0, g0, b0, p0), calc_radius(vf - dvf, eta0, rho0, g0, b0, p0)
    a_plus_rho, a_minus_rho = calc_radius(vf, eta0, rho0 + params['rho'][1], g0, b0, p0), calc_radius(vf, eta0, rho0 - params['rho'][1], g0, b0, p0)
    a_plus_b, a_minus_b = calc_radius(vf, eta0, rho0, g0, b0 + params['b'][1], p0), calc_radius(vf, eta0, rho0, g0, b0 - params['b'][1], p0)
    a_plus_p, a_minus_p = calc_radius(vf, eta0, rho0, g0, b0, p0 + params['p'][1]), calc_radius(vf, eta0, rho0, g0, b0, p0 - params['p'][1])

    da = np.sqrt(
        ((a_plus_eta - a_minus_eta)/2)**2 +
        ((a_plus_vf - a_minus_vf)/2)**2 +
        ((a_plus_rho - a_minus_rho)/2)**2 +
        ((a_plus_b - a_minus_b)/2)**2 +
        ((a_plus_p - a_minus_p)/2)**2
    )

    # Integer charge assignment
    n = round(q0 / e_accepted)
    if n < 1:
        n = 1

    e_calc = q0 / n
    # Uncertainty in e_calc = dq / n
    de_calc = dq / n

    error_pct = (e_calc - e_accepted) / e_accepted * 100

    results.append({
        'video': row['video'],
        'track_id': row['track_id'],
        'vf_mm_s': row['vf_mm_s'],
        'vr_mm_s': row['vr_mm_s'],
        'delta_vf_mm_s': dvf * 1e3,
        'delta_vr_mm_s': dvr * 1e3,
        'radius_um': a0 * 1e6,
        'delta_radius_um': da * 1e6,
        'charge_C': q0,
        'delta_charge_C': dq,
        'n': n,
        'e_calc': e_calc,
        'delta_e_calc': de_calc,
        'error_pct': error_pct,
        # Individual contributions to dq (for error budget)
        'dq_d': delta_q_components['d'],
        'dq_V': delta_q_components['V'],
        'dq_eta': delta_q_components['eta'],
        'dq_rho': delta_q_components['rho'],
        'dq_b': delta_q_components['b'],
        'dq_p': delta_q_components['p'],
        'dq_vf': delta_q_components['vf'],
        'dq_vr': delta_q_components['vr'],
        'dq_g': delta_q_components['g'],
    })

out = pd.DataFrame(results)
out.to_csv('millikan_final_data.csv', index=False)
print(f"\nSaved {len(out)} rows to millikan_final_data.csv")

# =====================================================================
# Summary statistics
# =====================================================================
print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")

print(f"\nElementary charge:")
e_vals = out['e_calc']
de_vals = out['delta_e_calc']
mean_e = e_vals.mean()
std_e = e_vals.std()
# Weighted mean (weight = 1/sigma^2)
w = 1.0 / de_vals**2
weighted_mean_e = np.average(e_vals, weights=w)
weighted_err = 1.0 / np.sqrt(w.sum())

print(f"  Unweighted mean: ({mean_e:.4e} ± {std_e:.4e}) C")
print(f"  Weighted mean:   ({weighted_mean_e:.4e} ± {weighted_err:.4e}) C")
print(f"  Accepted:         {e_accepted:.4e} C")
print(f"  Unweighted error: {(mean_e - e_accepted)/e_accepted*100:+.2f}%")
print(f"  Weighted error:   {(weighted_mean_e - e_accepted)/e_accepted*100:+.2f}%")

# Compare old vs new
print(f"\n{'='*70}")
print("COMPARISON: OLD vs NEW PARAMETERS")
print(f"{'='*70}")
old_E = 500.0 / 7.6e-3
new_E = V0 / d0
print(f"  Old E = 500/0.0076 = {old_E:.1f} V/m")
print(f"  New E = 500/0.0075 = {new_E:.1f} V/m  ({(new_E/old_E - 1)*100:+.2f}% change)")
print(f"  Old η = 1.820e-5,  New η = {eta0:.3e}  ({(eta0/1.82e-5 - 1)*100:+.2f}% change)")
print(f"  E↑ means q↓ (inversely proportional)")
print(f"  η↓ means smaller radius → q↓")

# Error budget (average across all measurements)
print(f"\n{'='*70}")
print("ERROR BUDGET (RMS contribution to δq across all measurements)")
print(f"{'='*70}")
budget_cols = ['dq_d', 'dq_V', 'dq_eta', 'dq_rho', 'dq_b', 'dq_p', 'dq_vf', 'dq_vr', 'dq_g']
budget_labels = ['d (plate sep)', 'V (voltage)', 'η (viscosity)', 'ρ (oil density)',
                 'b (Cunningham)', 'p (pressure)', 'vf (fall vel)', 'vr (rise vel)', 'g (gravity)']

# Compute fractional contributions
total_dq2 = sum(out[c]**2 for c in budget_cols)
for label, col in zip(budget_labels, budget_cols):
    frac = (out[col]**2 / total_dq2).mean() * 100
    rms = np.sqrt((out[col]**2).mean())
    print(f"  {label:20s}: {frac:5.1f}%  (RMS δq = {rms:.2e} C)")

# Threshold analysis with new values
print(f"\n{'='*70}")
print("MEASUREMENT QUALITY")
print(f"{'='*70}")
errs = out['error_pct'].abs()
for thresh in [1, 2, 3, 5, 10]:
    n_within = (errs < thresh).sum()
    print(f"  Within {thresh:2d}%: {n_within:3d}/{len(out)} ({100*n_within/len(out):.1f}%)")

# Outliers
outliers = out[errs > 10]
print(f"\nOutliers (|error| > 10%): {len(outliers)}")
for _, r in outliers.iterrows():
    print(f"  {r['video']} track {r['track_id']}: n={r['n']}, error={r['error_pct']:+.1f}%, "
          f"δq/q={r['delta_charge_C']/r['charge_C']*100:.1f}%")

print(f"\n{'='*70}")
print("PER-TRIAL SUMMARY")
print(f"{'='*70}")
for vid in out['video'].unique():
    sub = out[out['video'] == vid]
    w_sub = 1.0 / sub['delta_e_calc']**2
    wm = np.average(sub['e_calc'], weights=w_sub)
    we = 1.0 / np.sqrt(w_sub.sum())
    print(f"  {vid:20s}: N={len(sub):3d}, e = ({wm:.4e} ± {we:.4e}) C, "
          f"error = {(wm-e_accepted)/e_accepted*100:+.2f}%")
