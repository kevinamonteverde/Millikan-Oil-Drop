import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters (must match the simulation parameters)
params = {
	'V': 500.0,
	'd': 5.00e-3,
	'eta': 1.84e-5,
	'p': 1.0345e5,
	'rho': 8.800e2,
	'q_e': 1.609e-19,
	'b': 8.20e-3,
	'g': 9.7901,
}


def load_simulated_data():
	base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulated')
	perfect_path = os.path.join(base_dir, 'perfect_drops.csv')
	fall_path = os.path.join(base_dir, 'smeared_fall_velocities.csv')
	rise_path = os.path.join(base_dir, 'smeared_rise_velocities.csv')

	perfect = pd.read_csv(perfect_path)
	smeared_fall = pd.read_csv(fall_path)
	smeared_rise = pd.read_csv(rise_path)

	return perfect, smeared_fall, smeared_rise


def compute_radius_from_fall(fall_velocity):
	b = params['b']
	p = params['p']
	eta = params['eta']
	g = params['g']
	rho = params['rho']

	b_over_2p = b / (2 * p)
	inside = b_over_2p ** 2 + (9 * eta * fall_velocity) / (2 * g * rho)
	return np.sqrt(inside) - b_over_2p


def compute_charge_constant(radius):
	# charge constant as used in the simulation
	d = params['d']
	V = params['V']
	eta = params['eta']
	g = params['g']
	rho = params['rho']
	b = params['b']
	p = params['p']

	correction_factors = 1 + (b / (p * radius))
	prefac = (6 * math.pi * d / V)
	inner = (9 * eta ** 3) / (2 * g * rho * correction_factors ** 3)
	return prefac * np.sqrt(inner)


def estimate_qs(perfect, smeared_fall, smeared_rise):
	# Ensure drop_id is a column we can use as index
	if 'drop_id' in smeared_fall.columns:
		smeared_fall = smeared_fall.set_index('drop_id')
	if 'drop_id' in smeared_rise.columns:
		smeared_rise = smeared_rise.set_index('drop_id')

	records = []
	for _, drop in perfect.iterrows():
		drop_id = int(drop['id'])
		n_charges = int(drop['number_of_charges[e]'])

		for i in range(1, 11):
			fcol = f'measurement_{i}'
			try:
				fall = float(smeared_fall.loc[drop_id, fcol])
				rise = float(smeared_rise.loc[drop_id, fcol])
			except Exception:
				# Skip if measurement missing
				continue

			radius = compute_radius_from_fall(fall)
			charge_const = compute_charge_constant(radius)

			# invert the relation used in the simulation:
			# rise + fall = (n*q) / (charge_const * sqrt(fall))
			q_est = (rise + fall) * charge_const * np.sqrt(fall) / n_charges

			records.append({
				'drop_id': drop_id,
				'n_charges': n_charges,
				'measurement': i,
				'fall_velocity': fall,
				'rise_velocity': rise,
				'radius': radius,
				'charge_const': charge_const,
				'q_est[C]': q_est,
			})

	q_df = pd.DataFrame.from_records(records)
	return q_df


def summarize_qs(q_df):
	overall_mean = q_df['q_est[C]'].mean()
	overall_std = q_df['q_est[C]'].std(ddof=1)
	n = len(q_df)
	se = overall_std / np.sqrt(n) if n > 0 else np.nan

	per_drop = q_df.groupby('drop_id')['q_est[C]'].agg(['mean', 'std', 'count']).reset_index()

	summary = {
		'n_measurements': n,
		'mean_q[C]': overall_mean,
		'std_q[C]': overall_std,
		'stderr_q[C]': se,
		'relative_error_percent': 100.0 * (overall_mean - params['q_e']) / params['q_e'] if not np.isnan(overall_mean) else np.nan,
	}

	return summary, per_drop


def save_outputs(q_df, summary, per_drop):
	out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulated')
	os.makedirs(out_dir, exist_ok=True)
	q_df.to_csv(os.path.join(out_dir, 'q_estimates.csv'), index=False)
	pd.DataFrame([summary]).to_csv(os.path.join(out_dir, 'q_summary.csv'), index=False)
	per_drop.to_csv(os.path.join(out_dir, 'q_per_drop.csv'), index=False)

	# Save a diagnostic histogram
	try:
		plt.figure(figsize=(6, 4))
		plt.hist(q_df['q_est[C]'], bins=30)
		plt.axvline(params['q_e'], color='r', linestyle='--', label='input q_e')
		plt.xlabel('q estimate [C]')
		plt.ylabel('counts')
		plt.legend()
		plt.tight_layout()
		plt_path = os.path.join(out_dir, 'q_histogram.png')
		plt.savefig(plt_path)
		plt.close()
	except Exception:
		pass


def main():
	perfect, smeared_fall, smeared_rise = load_simulated_data()
	q_df = estimate_qs(perfect, smeared_fall, smeared_rise)
	summary, per_drop = summarize_qs(q_df)
	save_outputs(q_df, summary, per_drop)

	print('Part B analysis complete.')
	print('Overall estimate of q: {:.4e} C ± {:.4e} (SE)'.format(summary['mean_q[C]'], summary['stderr_q[C]']))
	print('Relative error vs input q_e: {:.2f}%'.format(summary['relative_error_percent']))


if __name__ == '__main__':
	main()

