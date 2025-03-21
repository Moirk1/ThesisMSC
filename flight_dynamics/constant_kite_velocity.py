import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

flight_data = pd.read_csv('2024_week_17_thu_03\\kite_power_production.csv')
timestamp = flight_data['epoch_ms'] - flight_data['epoch_ms'][0] 

# Columns of interest
columns_of_interest = ['V_a', 'v_eb_eE', 'v_eb_eN', 'psi', 'LoadFiltered']  # Update with actual column names

# Convert psi to radians if not already done in the dataset
if 'psi' in flight_data.columns:
    flight_data['psi'] = np.radians(flight_data['psi'])
if 'LoadFiltered' in flight_data.columns:
    flight_data['LoadFiltered'] = flight_data['LoadFiltered']* 9.80665

# Initialize dictionary to store block means for each column
block_means_dict = {}

# Loop through each column and compute block means
for col in columns_of_interest:
    values = flight_data[col].values  # Extract column values

    # Identify where values are NOT NaN
    valid_mask = ~np.isnan(values)

    # Find start and end indices of valid blocks
    diff = np.diff(valid_mask.astype(int))
    block_start_indices = np.where(diff == 1)[0] + 1  # NaN → valid transition
    block_end_indices = np.where(diff == -1)[0]  # valid → NaN transition

    # Handle edge cases where the first or last block extends to dataset boundaries
    if valid_mask[0]:  
        block_start_indices = np.insert(block_start_indices, 0, 0)
    if valid_mask[-1]:  
        block_end_indices = np.append(block_end_indices, len(values) - 1)

    # Compute means for each block
    block_means = [np.nanmean(values[start:end + 1]) for start, end in zip(block_start_indices, block_end_indices)]

    # Store results
    block_means_dict[col] = np.array(block_means)

# Print the means for each variable
#for col, means in block_means_dict.items():
    #print(f"Block means for {col}: {means}")


# Constants
rho = 1.225  # Air density (kg/m^3)
A = 3      # Kite area (m^2) Taken from Roland Schmehl lectures
CD0 = 0.004   # Minimum drag coefficient (Both these values are NACA 2412)
a = 0.008    # Drag coefficient scaling constant

# Store Ft results
Ft_array = []
# Store LoadFiltered means
load_filtered_means = []
# Loop through each valid block
for start, end in zip(block_start_indices, block_end_indices):
    # Compute mean of each column in the block
    mean_values = flight_data.iloc[start:end+1][columns_of_interest].mean()

    v_airspeed = mean_values['V_a']
    v_n = mean_values['v_eb_eN']
    v_e = mean_values['v_eb_eE']
    psi = mean_values['psi']
    load_filtered = mean_values['LoadFiltered'] 

    # Ground reference frame
    v_ground = np.sqrt(v_n**2 + v_e**2)
    v_w = v_ground - v_airspeed

    # Find velocity of kite
    v_kx = v_airspeed * np.sin(psi)
    v_ky = v_airspeed * np.cos(psi)

    # Compute apparent wind velocity components
    v_ax = v_w - v_kx
    v_ay = -v_ky

    # Compute apparent wind speed
    v_a = np.sqrt(v_ax**2 + v_ay**2)

    # Compute angle of attack
    alpha = np.arctan2(v_ax, -v_ay)  # In radians

    # Compute aerodynamic coefficients
    CL = 2 * np.pi * alpha  # Lift coefficient
    CD = CD0 + a * CL**2    # Drag coefficient

    # Compute forces
    L = 0.5 * CL * rho * A * v_a**2
    D = 0.5 * CD * rho * A * v_a**2

    # Compute tension force
    Ft = np.sqrt(L**2 + D**2)
    
    # Store result
    Ft_array.append(Ft)
    load_filtered_means.append(load_filtered)

# Convert to NumPy array
Ft_array = np.array(Ft_array)
load_filtered_means = np.array(load_filtered_means)

# Print results [IF I'M CORRECT THESE SHOULD BE GIVING OUT ROUGHLY THE RIGHT VALUES]
print("LoadFiltered means for each block:")
print(load_filtered_means)

print("\nTension Force (Ft) for each block:")
print(Ft_array)
