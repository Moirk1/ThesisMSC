import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

flight_data = pd.read_csv('2024_week_17_thu_03\\kite_power_production.csv')

# Time adjustment
timestamp = flight_data['epoch_ms'] - flight_data['epoch_ms'][0]

# Columns of interest
columns_of_interest = ['V_a', 'v_eb_eE', 'v_eb_eN', 'psi', 'LoadFiltered']

# Convert psi to radians
if 'psi' in flight_data.columns:
    flight_data['psi'] = np.radians(flight_data['psi'])
if 'LoadFiltered' in flight_data.columns:
    flight_data['LoadFiltered'] *= 9.80665  # Convert to Newtons

# Compute block means
block_means_dict = {}
for col in columns_of_interest:
    values = flight_data[col].values
    valid_mask = ~np.isnan(values)

    diff = np.diff(valid_mask.astype(int))
    block_start_indices = np.where(diff == 1)[0] + 1
    block_end_indices = np.where(diff == -1)[0]

    if valid_mask[0]:  
        block_start_indices = np.insert(block_start_indices, 0, 0)
    if valid_mask[-1]:  
        block_end_indices = np.append(block_end_indices, len(values) - 1)

    block_means = [np.nanmean(values[start:end + 1]) for start, end in zip(block_start_indices, block_end_indices)]
    block_means_dict[col] = np.array(block_means)

# Constants
rho = 1.225  # kg/m³
A = 3        # m² (kite area)
CD0 = 0.004  # Minimum drag coefficient (Taken from NACA 2412 airfoil)
a = 0.008    # Drag coefficient scaling factor

# Extract mean values for optimisation
block_data = []
load_filtered_means = []

for start, end in zip(block_start_indices, block_end_indices):
    mean_values = flight_data.iloc[start:end+1][columns_of_interest].mean()
    v_airspeed = mean_values['V_a']
    v_n = mean_values['v_eb_eN']
    v_e = mean_values['v_eb_eE']
    psi = mean_values['psi']
    load_filtered = mean_values['LoadFiltered']

    # Compute apparent wind velocity
    v_ground = np.sqrt(v_n**2 + v_e**2)
    v_w = v_ground - v_airspeed

    v_kx = v_airspeed * np.sin(psi)
    v_ky = v_airspeed * np.cos(psi)

    v_ax = v_w - v_kx
    v_ay = -v_ky
    v_a = np.sqrt(v_ax**2 + v_ay**2)

    alpha = np.abs(np.arctan2(v_ax, -v_ay))  # Angle of attack

    block_data.append((v_a, alpha, load_filtered))
    load_filtered_means.append(load_filtered)

# Convert lists to NumPy arrays
load_filtered_means = np.array(load_filtered_means)

# Optimisation function to find a single k across all blocks
def optimize_global_k(k):
    total_error = 0
    for v_a, alpha, load_filtered in block_data:
        CL = k * (2 * np.pi * alpha)  # Compute lift coefficient
        CD = CD0 + a * CL**2          # Compute drag coefficient
        
        L = 0.5 * CL * rho * A * v_a**2
        D = 0.5 * CD * rho * A * v_a**2

        Ft = np.sqrt(L**2 + D**2)
        total_error += abs(Ft - load_filtered)  # Sum absolute errors

    return total_error

# Optimise a single k across all blocks
result = minimize_scalar(optimize_global_k, bounds=(0.1, 10), method='bounded')

if result.success:
    k_opt = result.x
else:
    k_opt = 1  # Fallback if optimization fails

# Compute single optimised CL and CD
CL_opt = k_opt * (2 * np.pi * np.array([alpha for _, alpha, _ in block_data]))
CD_opt = CD0 + a * CL_opt**2

# Compute optimised forces
Ft_array = []
ratio_LD = []
for (v_a, alpha, load_filtered), CL in zip(block_data, CL_opt):
    L_opt = 0.5 * CL * rho * A * v_a**2
    D_opt = 0.5 * (CD0 + a * CL**2) * rho * A * v_a**2
    Ft_opt = np.sqrt(L_opt**2 + D_opt**2)
    ratio = L_opt/D_opt
    ratio_LD.append(ratio)
    Ft_array.append(Ft_opt)

Ft_array = np.array(Ft_array)
ratio_array = np.array(ratio_LD)
# Print results

print("Optimised k:", k_opt)
print("\nLoadFiltered means for each block:")
print(load_filtered_means)
print("\nOptimised Tension Force (Ft) for each block:")
print(Ft_array)
print("\nLift and Drag ratio")
print(ratio_array)


# Create a figure with multiple subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

# Plot Ft vs LoadFiltered on the first subplot
ax1.plot(range(len(Ft_array)), Ft_array, label='Ft (Calculated)', color='blue', linestyle='-', marker='o')
ax1.plot(range(len(load_filtered_means)), load_filtered_means, label='LoadFiltered (Sensor)', color='red', linestyle='--', marker='x')
#ax1.set_xlabel('Power Production Cycle')
ax1.set_ylabel('Tension Force (N)')
#ax1.set_title('Tension Force: Ft vs LoadFiltered')
ax1.legend()

# Plot CL and CD on the second subplot
ax2.plot(range(len(Ft_array)), CL_opt, label='CL', color='green', linestyle='-', marker='o')
ax2.plot(range(len(Ft_array)), CD_opt, label='CD', color='orange', linestyle='--', marker='x')
#ax2.set_xlabel('Power Production Cycle')
ax2.set_ylabel('Lift and Drag Coefficients')
#ax2.set_title('Lift and Drag Coefficients: CL vs CD')
ax2.legend()

# Plot L/D ratio on the third subplot
ax3.plot(range(len(Ft_array)), ratio_array, label='L/D Ratio', color='purple', linestyle='-', marker='o')
ax3.set_xlabel('Power Production Cycle')
ax3.set_ylabel('Lift to Drag Ratio')
#ax3.set_title('Lift to Drag Ratio (L/D)')
ax3.legend()

# Adjust layout to avoid overlap
#plt.tight_layout()
plt.savefig('inferclcd.pdf', format='pdf')
# Show the plots
plt.show()
