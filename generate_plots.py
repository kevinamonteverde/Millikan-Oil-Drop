#!/usr/bin/env python3
"""Generate all plots for the Millikan oil drop lab report."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

df = pd.read_csv('millikan_final_data.csv')
e_accepted = 1.602e-19

out = 'plots'
import os
os.makedirs(out, exist_ok=True)

# ============================================================
# 1. Histogram of calculated elementary charge values
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
e_vals = df['e_calc'] * 1e19  # scale to 10^-19 C
bins = np.linspace(1.45, 1.85, 30)
ax.hist(e_vals, bins=bins, color='steelblue', edgecolor='white', linewidth=0.5, alpha=0.85)
ax.axvline(1.602, color='red', linestyle='--', linewidth=1.5, label=f'Accepted $e$ = 1.602')
mean_e = df['e_calc'].mean() * 1e19
ax.axvline(mean_e, color='orange', linestyle='-', linewidth=1.5, label=f'Measured mean = {mean_e:.3f}')
ax.set_xlabel(r'Calculated $e$ ($\times 10^{-19}$ C)')
ax.set_ylabel('Number of Measurements')
ax.set_title('Distribution of Calculated Elementary Charge')
ax.legend()
fig.savefig(f'{out}/histogram_elementary_charge.png')
plt.close()

# ============================================================
# 2. Charge quantization plot: q vs n with lines q = n*e
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5.5))
colors_map = {'trial1drop1.mov': '#e41a1c', 'trial2.mov': '#377eb8',
              'trial3.mov': '#4daf4a', 'trial4.mov': '#984ea3'}
labels_map = {'trial1drop1.mov': 'Trial 1', 'trial2.mov': 'Trial 2',
              'trial3.mov': 'Trial 3', 'trial4.mov': 'Trial 4'}
for vid in df['video'].unique():
    sub = df[df['video'] == vid]
    ax.scatter(sub['n'], sub['charge_C'] * 1e19, s=25, alpha=0.7,
               color=colors_map.get(vid, 'gray'), label=labels_map.get(vid, vid),
               edgecolors='k', linewidth=0.3)
n_line = np.arange(0, 22)
ax.plot(n_line, n_line * e_accepted * 1e19, 'k--', linewidth=1, alpha=0.6, label='$q = n \\cdot e$')
ax.set_xlabel('Number of Elementary Charges ($n$)')
ax.set_ylabel(r'Measured Charge ($\times 10^{-19}$ C)')
ax.set_title('Charge Quantization')
ax.legend(loc='upper left', framealpha=0.9)
ax.set_xlim(0, 22)
ax.xaxis.set_major_locator(MultipleLocator(2))
fig.savefig(f'{out}/charge_quantization.png')
plt.close()

# ============================================================
# 3. Percent error distribution
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
errors = df['error_pct']
bins_err = np.linspace(-10, 25, 40)
ax.hist(errors, bins=bins_err, color='coral', edgecolor='white', linewidth=0.5, alpha=0.85)
ax.axvline(0, color='k', linestyle='-', linewidth=0.8)
ax.axvline(errors.mean(), color='blue', linestyle='--', linewidth=1.5,
           label=f'Mean error = {errors.mean():.2f}%')
ax.set_xlabel('Percent Error from Accepted $e$ (%)')
ax.set_ylabel('Number of Measurements')
ax.set_title('Distribution of Percent Error')
ax.legend()
fig.savefig(f'{out}/error_distribution.png')
plt.close()

# ============================================================
# 4. Fall velocity vs Rise velocity scatter
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))
for vid in df['video'].unique():
    sub = df[df['video'] == vid]
    ax.scatter(sub['vf_mm_s'], sub['vr_mm_s'], s=25, alpha=0.7,
               color=colors_map.get(vid, 'gray'), label=labels_map.get(vid, vid),
               edgecolors='k', linewidth=0.3)
ax.plot([0, 0.35], [0, 0.35], 'k--', alpha=0.3, linewidth=1, label='$v_f = v_r$')
ax.set_xlabel('Fall Velocity $v_f$ (mm/s)')
ax.set_ylabel('Rise Velocity $v_r$ (mm/s)')
ax.set_title('Fall vs. Rise Velocities')
ax.legend(loc='upper left', framealpha=0.9)
fig.savefig(f'{out}/velocity_scatter.png')
plt.close()

# ============================================================
# 5. Droplet radius distribution
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['radius_um'], bins=25, color='mediumpurple', edgecolor='white', linewidth=0.5, alpha=0.85)
ax.set_xlabel(r'Droplet Radius ($\mu$m)')
ax.set_ylabel('Number of Droplets')
ax.set_title('Distribution of Measured Droplet Radii')
fig.savefig(f'{out}/radius_distribution.png')
plt.close()

# ============================================================
# 6. Electron count (n) distribution
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
n_counts = df['n'].value_counts().sort_index()
ax.bar(n_counts.index, n_counts.values, color='teal', edgecolor='white', linewidth=0.5, alpha=0.85)
ax.set_xlabel('Number of Elementary Charges ($n$)')
ax.set_ylabel('Number of Droplets')
ax.set_title('Distribution of Electron Counts per Droplet')
ax.xaxis.set_major_locator(MultipleLocator(2))
fig.savefig(f'{out}/electron_count_distribution.png')
plt.close()

# ============================================================
# 7. Residuals: e_calc vs measurement index, colored by trial
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4.5))
for vid in df['video'].unique():
    sub = df[df['video'] == vid]
    ax.scatter(sub.index, sub['e_calc'] * 1e19, s=20, alpha=0.7,
               color=colors_map.get(vid, 'gray'), label=labels_map.get(vid, vid),
               edgecolors='k', linewidth=0.2)
ax.axhline(1.602, color='red', linestyle='--', linewidth=1.2, label='Accepted $e$')
ax.axhline(mean_e, color='orange', linestyle='-', linewidth=1.2, label=f'Mean = {mean_e:.4f}')
ax.fill_between(range(len(df)), 1.602 - 0.05, 1.602 + 0.05, color='red', alpha=0.07)
ax.set_xlabel('Measurement Index')
ax.set_ylabel(r'$e_{calc}$ ($\times 10^{-19}$ C)')
ax.set_title('Calculated $e$ per Measurement')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
fig.savefig(f'{out}/e_calc_per_measurement.png')
plt.close()

# ============================================================
# 8. Summary statistics by trial (bar chart with error bars)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
trial_stats = df.groupby('video')['e_calc'].agg(['mean', 'std', 'count'])
trial_stats.index = [labels_map.get(v, v) for v in trial_stats.index]
trial_stats['mean_scaled'] = trial_stats['mean'] * 1e19
trial_stats['std_scaled'] = trial_stats['std'] * 1e19
colors_bar = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
bars = ax.bar(trial_stats.index, trial_stats['mean_scaled'],
              yerr=trial_stats['std_scaled'], capsize=5,
              color=colors_bar[:len(trial_stats)], edgecolor='k', linewidth=0.5, alpha=0.85)
ax.axhline(1.602, color='red', linestyle='--', linewidth=1.2, label='Accepted $e$')
# Annotate counts
for i, (idx, row) in enumerate(trial_stats.iterrows()):
    ax.text(i, row['mean_scaled'] + row['std_scaled'] + 0.005,
            f'n={int(row["count"])}', ha='center', fontsize=9)
ax.set_ylabel(r'Mean $e_{calc}$ ($\times 10^{-19}$ C)')
ax.set_title('Elementary Charge by Trial')
ax.legend()
fig.savefig(f'{out}/e_by_trial.png')
plt.close()

# ============================================================
# 9. Sensitivity analysis comparison (if file exists)
# ============================================================
try:
    sens = pd.read_csv('sensitivity_results.csv')
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(sens))
    ax.bar(x, sens['n_valid'], color='steelblue', alpha=0.8, edgecolor='k', linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(sens['name'], rotation=30, ha='right')
    ax.set_ylabel('Number of Valid Measurements')
    ax.set_title('Parameter Sensitivity: Valid Measurements by Configuration')
    # Twin axis for error
    ax2 = ax.twinx()
    ax2.plot(list(x), sens['error_pct'].abs(), 'ro-', markersize=6, linewidth=1.5, label='|Error %|')
    ax2.set_ylabel('|Percent Error| (%)')
    ax2.legend(loc='upper left')
    fig.savefig(f'{out}/sensitivity_analysis.png')
    plt.close()
except Exception:
    pass

print("All plots saved to plots/ directory:")
for f in sorted(os.listdir(out)):
    print(f"  {f}")

# Print summary stats for the document
print(f"\n{'='*60}")
print("FINAL STATISTICS FOR METHODOLOGY DOCUMENT")
print(f"{'='*60}")
print(f"Total valid measurements: {len(df)}")
print(f"Measurements by trial:")
for vid in df['video'].unique():
    sub = df[df['video'] == vid]
    print(f"  {vid}: {len(sub)} measurements, mean error = {sub['error_pct'].mean():+.2f}%")
print(f"\nOverall e_calc:")
print(f"  Mean:   {df['e_calc'].mean():.6e} C")
print(f"  Std:    {df['e_calc'].std():.6e} C")
print(f"  Median: {df['e_calc'].median():.6e} C")
print(f"  Mean error: {df['error_pct'].mean():+.2f}%")
print(f"  Std of error: {df['error_pct'].std():.2f}%")
print(f"\nAccepted value: {e_accepted:.6e} C")
print(f"Percent error:  {(df['e_calc'].mean() - e_accepted)/e_accepted*100:+.2f}%")

within_1 = (df['error_pct'].abs() < 1).sum()
within_2 = (df['error_pct'].abs() < 2).sum()
within_3 = (df['error_pct'].abs() < 3).sum()
within_5 = (df['error_pct'].abs() < 5).sum()
within_10 = (df['error_pct'].abs() < 10).sum()
print(f"\nMeasurements within thresholds:")
print(f"  Within 1%:  {within_1}/{len(df)} ({100*within_1/len(df):.1f}%)")
print(f"  Within 2%:  {within_2}/{len(df)} ({100*within_2/len(df):.1f}%)")
print(f"  Within 3%:  {within_3}/{len(df)} ({100*within_3/len(df):.1f}%)")
print(f"  Within 5%:  {within_5}/{len(df)} ({100*within_5/len(df):.1f}%)")
print(f"  Within 10%: {within_10}/{len(df)} ({100*within_10/len(df):.1f}%)")

print(f"\nElectron count distribution:")
print(df['n'].value_counts().sort_index().to_string())

print(f"\nDroplet radius stats:")
print(f"  Mean:  {df['radius_um'].mean():.3f} um")
print(f"  Std:   {df['radius_um'].std():.3f} um")
print(f"  Range: {df['radius_um'].min():.3f} - {df['radius_um'].max():.3f} um")

print(f"\nVelocity stats:")
print(f"  vf: mean={df['vf_mm_s'].mean():.4f}, std={df['vf_mm_s'].std():.4f}, range=[{df['vf_mm_s'].min():.4f}, {df['vf_mm_s'].max():.4f}] mm/s")
print(f"  vr: mean={df['vr_mm_s'].mean():.4f}, std={df['vr_mm_s'].std():.4f}, range=[{df['vr_mm_s'].min():.4f}, {df['vr_mm_s'].max():.4f}] mm/s")

# Outlier analysis
outliers = df[df['error_pct'].abs() > 10]
print(f"\nOutliers (|error| > 10%): {len(outliers)}")
for _, row in outliers.iterrows():
    print(f"  {row['video']} track {row['track_id']}: n={row['n']}, error={row['error_pct']:+.1f}%, "
          f"vf={row['vf_mm_s']:.4f}, vr={row['vr_mm_s']:.4f}")
