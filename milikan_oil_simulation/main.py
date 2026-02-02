import numpy as np
import pandas as pd
import os 

params = {
    'V': 500.0,  # Voltage [V]
    'd': 5.00e-3,  # Capacitor separation [m] (note: table says 5.00 +/- 0.01 x 10^-3, use central value)
    'eta': 1.84e-5,  # Viscosity of air [Ns/m]
    'p': 1.0345e5,  # Barometric pressure [Pa]
    'rho': 8.800e2,  # Density of oil [kg/m^3]
    'q_e': 1.609e-19,  # Elementary charge [C]
    'b': 8.20e-3,  # b constant [Pa m]
    'g': 9.7901,  # Gravitational constant [m/s^2]
    'reticle_distance': 5.0e-4  # 0.5 mm [m]
}


def populate_perfect_drops():
    """
    Creates a DataFrame of ten random perfect drops.

    The fall time and number of elementary charges are both uniformly distributed random variables.

    Additionally, perfect drops have fall and rise velocities, which are calculated using helper functions.
    """
    num_drops = 10
    fall_times = np.random.uniform(9.5, 20.5, num_drops)
    num_charges = np.random.randint(1, 11, num_drops)

    fall_velocities = params['reticle_distance'] / fall_times

    b_over_2p = params['b'] / (2 * params['p'])
    radii = (
            np.sqrt(b_over_2p ** 2 + (9 * params['eta'] * fall_velocities) / (2 * params['g'] * params['rho'])) -
            b_over_2p
    )

    correction_factors = 1 + (params['b'] / (params['p'] * radii))
    charge_constants = (
            (6 * np.pi * params['d'] / params['V']) *
            np.sqrt((9 * params['eta'] ** 3) / (2 * params['g'] * params['rho'] * correction_factors ** 3))
    )
    rise_velocities = (
            (num_charges * params['q_e']) / (charge_constants * np.sqrt(fall_velocities)) -
            fall_velocities
    )

    perfect_drops_df = pd.DataFrame({
        'id': range(1, num_drops + 1),
        'fall_time[s]': fall_times,
        'number_of_charges[e]': num_charges,
        'fall_velocity[m/s]': fall_velocities,
        'rise_velocity[m/s]': rise_velocities,
        'radius': radii
    })

    return perfect_drops_df


def get_perfect_fall_velocities(perfect_drops_df):
    return perfect_drops_df[['id', 'fall_velocity[m/s]']]


def get_perfect_rise_velocities(perfect_drops_df):
    return perfect_drops_df[['id', 'rise_velocity[m/s]']]


def get_smeared_fall_velocities(perfect_fall_df):
    data = []
    for _, row in perfect_fall_df.iterrows():
        drop_id = row['id']
        fall_velo = row['fall_velocity[m/s]']
        sigma = 0.1 * fall_velo
        noise = np.random.normal(0, sigma, 10)
        smeared_velos = fall_velo + noise
        row_dict = {'drop_id': drop_id}
        for i, v in enumerate(smeared_velos, 1):
            row_dict[f'measurement_{i}'] = v
        data.append(row_dict)
    smeared_fall_df = pd.DataFrame(data).set_index('drop_id')
    return smeared_fall_df


def get_smeared_rise_velocities(perfect_rise_df):
    data = []
    for _, row in perfect_rise_df.iterrows():
        drop_id = row['id']
        rise_velo = row['rise_velocity[m/s]']
        sigma = 0.1 * rise_velo
        noise = np.random.normal(0, sigma, 10)
        smeared_velos = rise_velo + noise
        row_dict = {'drop_id': drop_id}
        for i, v in enumerate(smeared_velos, 1):
            row_dict[f'measurement_{i}'] = v
        data.append(row_dict)
    smeared_rise_df = pd.DataFrame(data).set_index('drop_id')
    return smeared_rise_df


def print_perfect_drops(df):
    print("Perfect Drops:")
    print(df.to_string(index=False))
    print()


def print_perfect_fall_velocities(df):
    print("Perfect Fall Velocities:")
    print(df.to_string(index=False))
    print()


def print_perfect_rise_velocities(df):
    print("Perfect Rise Velocities:")
    print(df.to_string(index=False))
    print()


def print_smeared_fall_velocities(df):
    print("Smeared Fall Velocities:")
    print(df.to_string())
    print()


def print_smeared_rise_velocities(df):
    print("Smeared Rise Velocities:")
    print(df.to_string())
    print()


# Generate the data
perfect_drops = populate_perfect_drops()
perfect_fall_velocities = get_perfect_fall_velocities(perfect_drops)
perfect_rise_velocities = get_perfect_rise_velocities(perfect_drops)
smeared_fall_velocities = get_smeared_fall_velocities(perfect_fall_velocities)
smeared_rise_velocities = get_smeared_rise_velocities(perfect_rise_velocities)

# Example usage of print functions (uncomment to print)
# print_perfect_drops(perfect_drops)
# print_perfect_fall_velocities(perfect_fall_velocities)
# print_perfect_rise_velocities(perfect_rise_velocities)
print_smeared_fall_velocities(smeared_fall_velocities)
print_smeared_rise_velocities(smeared_rise_velocities)

out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulated')
os.makedirs(out_dir, exist_ok=True)

perfect_drops.to_csv(os.path.join(out_dir, 'perfect_drops.csv'), index=False)
smeared_fall_velocities.reset_index().to_csv(os.path.join(out_dir, 'smeared_fall_velocities.csv'), index=False)
smeared_rise_velocities.reset_index().to_csv(os.path.join(out_dir, 'smeared_rise_velocities.csv'), index=False)