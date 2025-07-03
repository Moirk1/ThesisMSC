#Import packages
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hipersim import MannTurbulenceField
import pandas as pd
from scipy.signal import coherence
import os
import seaborn as sns

#Import function files
from functions_spectrum import *
from turb_functions import *
from sampling_functions import *
from aerodynamics_functions import *
from butterworth_function import *




params_1 = {
        "U_mean": 12.5,
        "alpha_epsilon": 0.0456825588131349,
        "L":68.05627895300304		,
        "Gamma": 2.438897937014391,
        "desired_TI": 0.0707,
        "Nxyz" : (7500, 41, 41),
        "dxyz" : (1.0, 8.0, 8.0),
    }


"""
params_2 =   {
        "U_mean":12.5,
        "alpha_epsilon":0.0285661627136685,
        "L":44.26827113634448	,
        "Gamma": 4.53955251367095	,
        "desired_TI": 0.0678,
        "Nxyz" : (7500, 41, 41),
        "dxyz" : (1.0, 8.0, 8.0),
    }


params_3 =    {
        "U_mean":12.5,
        "alpha_epsilon":0.0605574743713997,
        "L":40.24832787882204	,
        "Gamma": 3.817364053975576	,
        "desired_TI": 0.0881,
        "Nxyz" : (7500, 41, 41),
        "dxyz" : (1.0, 8.0, 8.0),
    }

params_4 =    {
        "U_mean":12.5,
        "alpha_epsilon":0.0605574743713997,
        "L":40.24832787882204	,
        "Gamma": 3.817364053975576	,
        "desired_TI": 0.15,
        "Nxyz" : (7500, 41, 41),
        "dxyz" : (1.0, 8.0, 8.0),
    }
"""
sim_time = 900
dt = 0.1
#Umean = 12.5
# Generate the three boxes
data1_u, data1_v, data1_w, (DX1, DY1, DZ1) = generate_mann_box(
    params_1["U_mean"], params_1["desired_TI"], params_1["alpha_epsilon"], params_1["L"], params_1["Gamma"], params_1["Nxyz"], params_1["dxyz"]
)
"""
data2_u, data2_v, data2_w, (DX2, DY2, DZ2) = generate_mann_box(
    params_2["U_mean"], params_2["desired_TI"], params_2["alpha_epsilon"], params_2["L"], params_2["Gamma"], params_2["Nxyz"], params_2["dxyz"]
)
data3_u, data3_v, data3_w, (DX3, DY3, DZ3) = generate_mann_box(
   params_3["U_mean"], params_3["desired_TI"], params_3["alpha_epsilon"], params_3["L"], params_3["Gamma"], params_3["Nxyz"], params_3["dxyz"]
)

data4_u, data4_v, data4_w, (DX4, DY4, DZ4) = generate_mann_box_scaled(
   params_4["U_mean"], params_4["desired_TI"], params_4["alpha_epsilon"], params_4["L"], params_4["Gamma"], params_4["Nxyz"], params_4["dxyz"]
)
"""

# Now you have three turbulence boxes saved in variables data1_*, data2_*, data3_*
Nx, Ny, Nz = data1_u.shape
print("Box 1 shape:", data1_u.shape)
#print("Box 2 shape:", data2_u.shape)
#print("Box 3 shape:", data3_u.shape)
#print("Box 4 shape:", data4_u.shape)
print(DX1, DY1, DZ1)




# --- Sampling  of Turbulence Field ---
#Helical Sampling

# Helical sampling for each box
samples1, coords_grid1, coords_phys1, Urel_1 = helical_sample_velocity_field_physical_moving_box(
    data1_u, data1_v, data1_w, U_box=params_1["U_mean"],
    radius_m=60,
    T_loop=9.23,
    total_time=900,
    DX=DX1, DY=DY1, DZ=DZ1
)
"""
samples2, coords_grid2, coords_phys2, Urel_2 = helical_sample_velocity_field_physical_moving_box(
    data2_u, data2_v, data2_w, U_box=params_2["U_mean"],
    radius_m=60,
    T_loop=9.23,
    total_time=900,
    DX=DX2, DY=DY2, DZ=DZ2
)

samples3, coords_grid3, coords_phys3, Urel_3 = helical_sample_velocity_field_physical_moving_box(
    data3_u, data3_v, data3_w, U_box=params_3["U_mean"],
    radius_m=60,
    T_loop=9.23,
    total_time=900,
    DX=DX3, DY=DY3, DZ=DZ3
)

samples4, coords_grid4, coords_phys4, Urel_4 = helical_sample_velocity_field_physical_moving_box(
    data4_u, data4_v, data4_w, U_box=params_4["U_mean"],
    radius_m=60,
    T_loop=9.23,
    total_time=900,
    DX=DX4, DY=DY4, DZ=DZ4
)
"""

# Assuming you have samples1, samples2, samples3
# and coords_phys1, coords_phys2, coords_phys3
# Also assuming U_mean1, U_mean2, U_mean3, Ny, Nz for each box

# You might want to extract Ny, Nz from each box's shape:
Ny1, Nz1 = data1_u.shape[1], data1_u.shape[2]
#Ny2, Nz2 = data2_u.shape[1], data2_u.shape[2]
#Ny3, Nz3 = data3_u.shape[1], data3_u.shape[2]
#Ny4, Nz4 = data4_u.shape[1], data4_u.shape[2]

# Define simulation time vector for each (assuming dt is global)
t1 = np.linspace(0, dt * len(samples1), len(samples1), endpoint=False)
#t2 = np.linspace(0, dt * len(samples2), len(samples2), endpoint=False)
#t3 = np.linspace(0, dt * len(samples3), len(samples3), endpoint=False)
#t4 = np.linspace(0, dt * len(samples4), len(samples4), endpoint=False)

def run_aero_analysis(samples, coords_phys, U_mean, Ny, Nz, DY, DZ):
    x, y, z = coords_phys[:, 0], coords_phys[:, 1], coords_phys[:, 2]
    y_max = np.max(y)
    y_min = np.min(y)
    print(f"y max: {y_max:.2f} meters")
    print(f"y min: {y_min:.2f} meters")

    # Velocity and acceleration as before
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    dz = np.gradient(z, dt)
    v_global = np.vstack((dx, dy, dz)).T

    dvx = np.gradient(dx, dt)
    dvy = np.gradient(dy, dt)
    dvz = np.gradient(dz, dt)
    a_global = np.vstack((dvx, dvy, dvz)).T

    # Compute phi_rad if needed (optional, depends on your aero function)
    y_offset = (Ny * DY) / 2
    z_offset = (Nz * DZ) / 2
    phi_rad = np.arctan2(z - z_offset, y - y_offset)
    phi_deg = np.degrees(phi_rad)

    # Run aero without turbulence to get pitch reference
    samples_noturb = np.zeros_like(samples)
    results_noturb = compute_aerodynamics(
        t1, x, y, z, v_global, samples_noturb,
        U_mean=U_mean, rho=1.225, A=2.982, CD0=0.004, a=0.04,
        DY=DY, DZ=DZ, Ny=Ny, Nz=Nz,
        fix_aoa_deg=6.0  
    )
    pitch_deg = results_noturb['awa_deg'] - results_noturb['aoa_deg']
    print(pitch_deg)
    # Run aero with turbulence
    results_turb = compute_aerodynamics(
        t1, x, y, z, v_global, samples,
        U_mean=U_mean, rho=1.225, A=2.982, CD0=0.004, a=0.04,
        DY=DY, DZ=DZ, Ny=Ny, Nz=Nz,
        pitch_reference=pitch_deg  
    )
    return results_turb, results_noturb

fs = 1 / dt  # Sampling frequency = 10 Hz
cutoff = 0.3
n_per_decade = 20
filtered_samples1 = butter_lowpass_filter(samples1, cutoff, fs)
#filtered_samples2 = butter_lowpass_filter(samples2, cutoff, fs)
#filtered_samples3 = butter_lowpass_filter(samples3, cutoff, fs)
#filtered_samples4 = butter_lowpass_filter(samples4, cutoff, fs)

spectral_accel_9_5, accel_freqs_9_5, accel_psd_9_5 = compute_spectral_acceleration_time(filtered_samples1[:,0], fs)
#spectral_accel_12_5, accel_freqs_12_5, accel_psd_12_5 = compute_spectral_acceleration_time(filtered_samples2[:,0], fs)
#spectral_accel_15_5, accel_freqs_15_5, accel_psd_15_5 = compute_spectral_acceleration_time(filtered_samples3[:,0], fs)
#spectral_accel_TI, accel_freqs_TI, accel_psd_TI = compute_spectral_acceleration_time(filtered_samples4[:,0], fs)

#smoothed_freqs_9_5_accel, smoothed_spectrum_9_5_accel = smooth_criminal( accel_psd_9_5, accel_freqs_9_5, n_per_decade)
#smoothed_freqs_12_5_accel, smoothed_spectrum_12_5_accel = smooth_criminal( accel_psd_12_5, accel_freqs_12_5, n_per_decade)
#smoothed_freqs_15_5_accel, smoothed_spectrum_15_5_accel = smooth_criminal( accel_psd_15_5, accel_freqs_15_5, n_per_decade)
#smoothed_freqs_TI_accel, smoothed_spectrum_TI_accel = smooth_criminal( accel_psd_TI, accel_freqs_TI, n_per_decade)

# Trimmed data
data_9_5 = spectral_accel_9_5[10:-10]
#data_12_5 = spectral_accel_12_5[10:-10]
#data_15_5 = spectral_accel_15_5[10:-10]
#data_TI = spectral_accel_TI[10:-10]

print(len(data_9_5))
#print(len(data_12_5))
#print(len(data_15_5))
# Calculate mean and std
mu_9_5, sigma_9_5 = np.mean(data_9_5), np.std(data_9_5)
#mu_12_5, sigma_12_5 = np.mean(data_12_5), np.std(data_12_5)
#mu_15_5, sigma_15_5 = np.mean(data_15_5), np.std(data_15_5)
#mu_TI, sigma_TI = np.mean(data_TI), np.std(data_TI)

print(mu_9_5, sigma_9_5)
#print(mu_12_5, sigma_12_5)
#print(mu_15_5, sigma_15_5)
#print(mu_TI, sigma_TI)
# Create x range
x_gauss = np.arange(-3.25, 3.25, 0.05)


pdf_9_5 = gauss_pdf(x_gauss, mu_9_5, sigma_9_5)
#pdf_12_5 = gauss_pdf(x_gauss, mu_12_5, sigma_12_5)
#pdf_15_5 = gauss_pdf(x_gauss, mu_15_5, sigma_15_5)
#pdf_TI = gauss_pdf(x_gauss, mu_TI, sigma_TI)


area_9_5 = np.trapz(pdf_9_5, x_gauss)
#area_12_5 = np.trapz(pdf_12_5, x_gauss)
#area_15_5 = np.trapz(pdf_15_5, x_gauss)
#area_TI = np.trapz(pdf_TI, x_gauss)

# Normalize by peak
pdf_9_5 /= np.max(pdf_9_5)
#pdf_12_5 /= np.max(pdf_12_5)
#pdf_15_5 /= np.max(pdf_15_5)

print("Area under Gaussian 9.5 m/s:", area_9_5)
#print("Area under Gaussian 12.5 m/s:", area_12_5)
#print("Area under Gaussian 15.5 m/s:", area_15_5)
#print("Area under Gaussian TI:", area_TI)

"""
colors = ['#0072B2',  # Blue
          '#E69F00',  # Orange
          '#009E73', #Green
          '#CC79A7', # Pink
          '#D55E00',  # Vermillion
          '#F0E442']  # Yellow
# Get the directory where your script is
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the existing 'TI_comparison' folder
figures_dir = os.path.join(script_dir, 'TI_comparison')
combined_min = min(data_9_5.min(), data_12_5.min(), data_15_5.min())#, data_TI.min())
combined_max = max(data_9_5.max(), data_12_5.max(), data_15_5.max())#, data_TI.max())

# Define bins with 40 intervals (or change as needed)
bins = np.linspace(combined_min, combined_max, 41)
bin_width = bins[1] - bins[0]
print(f"Bin width: {bin_width:.4f} m/s²")

# Compute histograms
counts_9_5, bins_9_5 = np.histogram(data_9_5, bins=bins)
#counts_12_5, bins_12_5 = np.histogram(data_12_5, bins=bins)
#counts_15_5, bins_15_5 = np.histogram(data_15_5, bins=bins)
#counts_TI, bins_TI = np.histogram(data_TI, bins=bins)

# Compute bin centers (same for all)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Plot as line graph with markers
plt.figure(figsize=(10, 5))

plt.plot(bin_centers, counts_9_5, color=colors[0], alpha=0.9, label='TI = 0.0493', linewidth=2, marker='o')
plt.plot(bin_centers, counts_12_5, color=colors[1], alpha=0.9, label='TI = 0.0678', linestyle='--', linewidth=2, marker='s')
plt.plot(bin_centers, counts_15_5, color=colors[2], alpha=0.9, label='TI = 0.0881', linestyle=':', linewidth=2, marker='^')
plt.plot(bin_centers, counts_TI, color=colors[3], alpha=0.9, label='TI = 0.1500', linestyle='-.', linewidth=2, marker='d')

plt.xlabel('Acceleration [m/s²]', fontsize=16)
plt.ylabel('Count [-]', fontsize=16)
plt.legend(fontsize=14, loc='upper right')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()

filename = 'hist_accel_trimmed_line.pdf'
full_path = os.path.join(figures_dir, filename)
plt.savefig(full_path, format='pdf', bbox_inches='tight')
plt.close()

"""
# Run for all three boxes
results_turb1, results_noturb1 = run_aero_analysis(filtered_samples1, coords_phys1, params_1["U_mean"], Ny1, Nz1, DY1, DZ1)
#results_turb2, results_noturb2 = run_aero_analysis(filtered_samples2, coords_phys2, params_2["U_mean"], Ny2, Nz2, DY2, DZ2)
#results_turb3, results_noturb3 = run_aero_analysis(filtered_samples3, coords_phys3, params_3["U_mean"], Ny3, Nz3, DY3, DZ3)
#results_turb4, results_noturb4 = run_aero_analysis(filtered_samples4, coords_phys4, params_4["U_mean"], Ny4, Nz4, DY4, DZ4)

# Define a list of results for each turbulent case
results_turb_list = [results_turb1] #results_turb2, results_turb3, results_turb4]

# Loop through each case and find exceedances
for i, results_turb in enumerate(results_turb_list, start=1):
    print(f"\n--- Exceedance Events for Case {i} ---")
    aoa = results_turb['aoa_deg']
    
    # Find exceedances (you can adjust threshold and min_duration as needed)
    events = find_aoa_exceedances(aoa, dt, threshold=10, min_duration=2)
    
    if events:
        for idx, (start, end) in enumerate(events):
            print(f"Event {idx+1}: AoA > 10 degrees from {start:.2f}s to {end:.2f}s (duration {end-start:.2f}s)")
    else:
        print("No exceedance events found.")


# --- Plotting ---
# Extract y, z coordinates of the helix
y = coords_phys1[:, 1]
z = coords_phys1[:, 2]

# Calculate offsets to center around helix axis
y_offset = (Ny1 * DY1) / 2
z_offset = (Nz1 * DZ1) / 2

# Compute circumferential angle in radians (-pi to pi)
phi_rad = np.arctan2(z - z_offset, y - y_offset)

# Convert to degrees [0, 360)
phi_deg = np.degrees(phi_rad) % 360

# Forces for each case
Fx1, Fy1, Fz1 = results_turb1['F_aero_vec'][:, 0], results_turb1['F_aero_vec'][:, 1], results_turb1['F_aero_vec'][:, 2]
#Fx2, Fy2, Fz2 = results_turb2['F_aero_vec'][:, 0], results_turb2['F_aero_vec'][:, 1], results_turb2['F_aero_vec'][:, 2]
#Fx3, Fy3, Fz3 = results_turb3['F_aero_vec'][:, 0], results_turb3['F_aero_vec'][:, 1], results_turb3['F_aero_vec'][:, 2]
#Fx4, Fy4, Fz4 = results_turb4['F_aero_vec'][:, 0], results_turb4['F_aero_vec'][:, 1], results_turb4['F_aero_vec'][:, 2]
# Combine for looping
cases = [(Fx1, Fy1, Fz1, '12.5 m/s')]
         #(Fx2, Fy2, Fz2, 'TI = 0.0678'),
         #(Fx3, Fy3, Fz3, 'TI = 0.0881'),
         #(Fx4, Fy4, Fz4, 'TI = 0.1500')]





colors = ['#0072B2',  # Blue
          '#E69F00',  # Orange
          '#009E73', #Green
          '#CC79A7', # Pink
          '#D55E00',  # Vermillion
          '#F0E442']  # Yellow
# Get the directory where your script is
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the existing 'TI_comparison' folder
figures_dir = os.path.join(script_dir, 'Steady_vs_Unsteady')
t = np.linspace(0, dt * len(samples1), len(samples1), endpoint=False)

# Example 1: Va_z vs AoA
va_z = results_turb1['v_apparent_vec'][:, 2]
aoa = results_turb1['aoa_deg']
corr_1 = np.corrcoef(va_z, aoa)[0, 1]
print(f"Correlation (Va_z vs AoA): {corr_1:.3f}")

# Example 4: Vr_x vs Va_z
vr_x = results_turb1['v_wind_NED'][:, 0]
corr_4 = np.corrcoef(vr_x, va_z)[0, 1]
print(f"Correlation (Vr_x vs Va_z): {corr_4:.3f}")

plt.figure(figsize=(10, 5))
# Plot in reverse order for flipped plot layering
plt.scatter(results_turb1['v_apparent_vec'][:,2], results_turb1['aoa_deg'],
         color=colors[2], alpha = 0.6)

plt.xlabel('$V_a$ [Z body] [m/s]', fontsize=16)
plt.ylabel('Angle of Attack [deg]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'aoa_va_12_5.pdf'), format='pdf')
plt.close()

plt.figure(figsize=(10, 5))
# Plot in reverse order for flipped plot layering
plt.scatter(results_turb1['v_apparent_vec'][:,0], results_turb1['aoa_deg'],
         color=colors[0], alpha = 0.6)

plt.xlabel('$V_a$ [X body] [m/s]', fontsize=16)
plt.ylabel('Angle of Attack [deg]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'aoa_vax_12_5.pdf'), format='pdf')
plt.close()

plt.figure(figsize=(10, 5))
# Plot in reverse order for flipped plot layering
plt.scatter(results_turb1['v_apparent_vec'][:,1], results_turb1['aoa_deg'],
         color=colors[1], alpha = 0.6)

plt.xlabel('$V_a$ [Y body] [m/s]', fontsize=16)
plt.ylabel('Angle of Attack [deg]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'aoa_vay_12_5.pdf'), format='pdf')
plt.close()



plt.figure(figsize=(10, 5))
# Plot in reverse order for flipped plot layering
plt.scatter(results_turb1['v_wind_NED'][:,0], results_turb1['v_apparent_vec'][:,2],
         color=colors[0], alpha = 0.6)

plt.xlabel('$V_r$ [X NED] [m/s]', fontsize=16)
plt.ylabel('$V_a$ [Z body] [m/s]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'v_wind_va_12_5.pdf'), format='pdf')
plt.close()

fig, ax = plt.subplots(figsize=(10, 5))  # use ax explicitly

# Plot in reverse order for flipped plot layering
ax.scatter(results_turb1['v_kite_NED'][:, 0],
           results_turb1['v_apparent_vec'][:, 2],
           color=colors[1], alpha=0.6)

# Disable scientific notation for both axes
ax.ticklabel_format(style='plain', axis='both', useOffset=False)

# Labeling
ax.set_xlabel('$V_k$ [X NED] [m/s]', fontsize=16)
ax.set_ylabel('$V_a$ [Z body] [m/s]', fontsize=16)
ax.tick_params(axis='both', labelsize=14)

ax.grid(True, linestyle=':', alpha=0.7, color='gray')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'v_kite_va_12_5.pdf'), format='pdf')
plt.close()


plt.figure(figsize=(10, 5))
# Plot in reverse order for flipped plot layering
plt.scatter(results_turb1['v_wind_NED'][:,1], results_turb1['v_apparent_vec'][:,2],
         color=colors[0], alpha = 0.6)

plt.xlabel('$V_{wind}$ [Y NED] [m/s]', fontsize=16)
plt.ylabel('$V_a$ [Z body] [m/s]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'v_windy_va_12_5.pdf'), format='pdf')
plt.close()


plt.figure(figsize=(10, 5))
# Plot in reverse order for flipped plot layering
plt.scatter(results_turb1['v_wind_NED'][:,2], results_turb1['v_apparent_vec'][:,2],
         color=colors[0], alpha = 0.6)

plt.xlabel('$V_{wind}$ [Z NED] [m/s]', fontsize=16)
plt.ylabel('$V_a$ [Z body] [m/s]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'v_windz_va_12_5.pdf'), format='pdf')
plt.close()



fig, ax1 = plt.subplots(figsize=(10, 5))

# First axis (left) — AoA
color1 = colors[0]
ax1.plot(t, results_turb1['v_wind_NED'][:,0], color=color1, alpha=0.6, label='V_wind')
ax1.set_xlabel('Time [s]', fontsize=16)
ax1.set_ylabel('V_wind [m/s]', fontsize=16, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle=':', alpha=0.7, color='gray')

# Second axis (right) — Airspeed
ax2 = ax1.twinx()
color2 = colors[1] if len(colors) > 1 else 'darkgreen'
ax2.plot(t, -1*results_turb1['v_apparent_vec'][:,2], color=color2, alpha=0.6, label='Airspeed')
ax2.set_ylabel('$V_a$ [m/s]', fontsize=16, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# Optional: Add legends
# You’ll need to combine them manually if you want both in one legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=12)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'time_va_aoa_12_5.pdf'), format='pdf')
plt.close()


"""
fig, axs = plt.subplots(1, 2, figsize=(18,6), subplot_kw={'projection': 'polar'}, constrained_layout=True)

thetagrids = np.arange(0, 360, 45)  # 0°, 45°, 90°, ...
rgrids = [-3000, -2500, -2000, -1500, -1000, -500, 0, 500, 1000]             # Example radial ticks

for i, (ax, (Fx, Fy, Fz, label)) in enumerate(zip(axs, cases)):
    ax.plot(phi_rad, Fy, '.', alpha=0.3, label=r'$F_y$', color=colors[0])
    ax.plot(phi_rad, Fz, '.', alpha=0.3, label=r'$F_z$', color=colors[1])
    ax.plot(phi_rad, Fx, '.', alpha=0.3, label=r'$F_x$', color=colors[2])

    ax.set_theta_zero_location('W')
    ax.set_theta_direction(-1)
    ax.set_title(label, fontsize=14)
    ax.set_rlabel_position(135)

    ax.set_thetagrids(thetagrids, fontsize=14)
    ax.set_rgrids(rgrids, fontsize=14)

    # Set zorder for ticks
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_zorder(10)

    

# Legend
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize=14)

# Save
plt.savefig(os.path.join(figures_dir, 'circumferential_angle_transients_15_5.pdf'), format='pdf')
plt.close()

for i, (Fx, Fy, Fz, label) in enumerate(cases):
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={'projection': 'polar'}, constrained_layout=True)

    # Plot
    ax.plot(phi_rad, Fz, '.', alpha=0.3, color=colors[i])

    # Polar formatting
    ax.set_theta_zero_location('W')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)

    # Grids and labels
    thetagrids = np.arange(0, 360, 45)
    rgrids = [-7000, -6500, -6000, -5500, -5000, -4500, -4000, -3500, -3000]

    ax.set_thetagrids(thetagrids, fontsize=14)
    ax.set_rgrids(rgrids, fontsize=14)

    # Title and legend
    #ax.set_title(f'Fz vs. Circumferential Angle\n{label}', fontsize=14)
    #ax.legend([label], loc='upper right', fontsize=12)

    # Save
    filename = f'Fz_circumferential_{label.replace(" ", "_")}.pdf'
    plt.savefig(os.path.join(figures_dir, filename), format='pdf')
    plt.close()




plt.figure(figsize=(10, 5))
#plt.style.use('tableau-colorblind10')
plt.plot(t, results_turb3['airspeed'], label='Unfiltered', linestyle='-')   # solid
#plt.plot(t, results_turb2['airspeed'], label='Filtered', linestyle='--')  # dashed
#plt.plot(t, results_turb3['airspeed'], label='15.5 m/s', linestyle=':')   # dotted
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Airspeed [m/s]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()


plt.figure(figsize=(10, 5))
#plt.style.use('tableau-colorblind10')

plt.plot(t, results_turb3['awa_deg'], label='Unfiltered', linestyle='-')
#plt.plot(t, results_turb2['awa_deg'], label='Filtered', linestyle='--')
#plt.plot(t, results_turb3['awa_deg'], label='15.5 m/s', linestyle=':')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Apparent Wind Angle [deg]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()

plt.figure()
plt.plot(t, results_turb3['lift_coeff'], label='Unfiltered', linestyle='-')
#plt.plot(t, results_turb2['awa_deg'], label='Filtered', linestyle='--')
#plt.plot(t, results_turb3['awa_deg'], label='15.5 m/s', linestyle=':')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('$C_L$', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()

aoa_cases = [
    results_turb1['aoa_deg'],
    results_turb2['aoa_deg'],
    results_turb3['aoa_deg'],
    results_turb4['aoa_deg']
]

for i, (aoa, (_, _, _, label)) in enumerate(zip(aoa_cases, cases)):
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={'projection': 'polar'}, constrained_layout=True)

    ax.plot(phi_rad, aoa, '.', alpha=0.3, color=colors[i])

    ax.set_theta_zero_location('W')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)

    thetagrids = np.arange(0, 360, 45)
    rgrids = [-2, 0, 2, 4, 6, 8, 10, 12, 14]  # Adjust based on AoA range

    ax.set_thetagrids(thetagrids, fontsize=14)
    ax.set_rgrids(rgrids, fontsize=14)

    #ax.legend([label], loc='upper right', fontsize=12)

    filename = f'aoa_circumferential_{label.replace(" ", "_")}.pdf'
    plt.savefig(os.path.join(figures_dir, filename), format='pdf')
    plt.close()







fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8,8))
ax.scatter(phi_rad, results_turb1['aoa_deg'], s=10, alpha=0.4, label = 'Unfiltered')
ax.scatter(phi_rad, results_turb2['aoa_deg'], s=10, alpha=0.4, label = 'Filtered')
ax.set_theta_zero_location('W')
ax.set_theta_direction(-1)
ax.set_rlabel_position(135)
# Increase label sizes
ax.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join(figures_dir, 'circumferential_aoa_transients_15_5.pdf'), format='pdf')
plt.close()



# 1) Angle of Attack plot
plt.figure(figsize=(10, 5))
# Plot in reverse order for flipped plot layering
plt.plot(t, results_turb4['aoa_deg'], label='TI = 0.1500', linestyle='-.', color=colors[3], linewidth=2.0)
plt.plot(t, results_turb3['aoa_deg'], label='TI = 0.0881', linestyle=':', color=colors[2], linewidth=2.0)
plt.plot(t, results_turb1['airspeed'], label='TI = 0.0678', linestyle='--', color=colors[1], linewidth=2.0,)
plt.plot(t, results_turb1['aoa_deg'], label='TI = 0.0493', linestyle='-', color=colors[0], linewidth=2.0, alpha = 0.8)

plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Angle of Attack [deg]', fontsize=16)

# Rearrange legend order to original (Steady, 9.5, 12.5, 15.5)
handles, labels = plt.gca().get_legend_handles_labels()
order = [3, 2, 1, 0]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
           loc='upper center',
           bbox_to_anchor=(0.5, 1.15),
           ncol=4,
           fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'aoa_TI.pdf'), format='pdf')
plt.close()

# 2) L/D ratio plot
plt.figure(figsize=(10, 5))
ld_ratio_turb4 = results_turb4['lift'] / results_turb4['drag']
ld_ratio_turb1 = results_turb1['lift'] / results_turb1['drag']
ld_ratio_turb2 = results_turb2['lift'] / results_turb2['drag']
ld_ratio_turb3 = results_turb3['lift'] / results_turb3['drag']

plt.plot(t, ld_ratio_turb4, label='TI = 0.1500', linestyle='-.', color=colors[3], linewidth=2.0)
plt.plot(t, ld_ratio_turb3, label='TI = 0.0881', linestyle=':', color=colors[2], linewidth=2.0)
plt.plot(t, ld_ratio_turb2, label='TI = 0.0678', linestyle='--', color=colors[1], linewidth=2.0)
plt.plot(t, ld_ratio_turb1, label='TI = 0.0493', alpha = 0.8, linestyle='-', color=colors[0], linewidth=2.0)

plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('L/D [-]', fontsize=16)

handles, labels = plt.gca().get_legend_handles_labels()
# Rearrange handles and labels to original legend order:
order = [3, 2, 1, 0]  # indices to reorder back to 9.5, 12.5, 15.5
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
           loc='upper center',
           bbox_to_anchor=(0.5, 1.15),
           ncol=4,
           fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'LDratio_TI.pdf'), format='pdf')
plt.close()

# 3) Fx aerodynamic force vector plot
plt.figure(figsize=(10, 5))
plt.plot(t, results_turb4['F_aero_vec'][:, 0]/1000, label='TI = 0.1500', linestyle='-.', color=colors[3], linewidth=2.0)
plt.plot(t, results_turb3['F_aero_vec'][:, 0]/1000, label='TI = 0.0881', linestyle=':', color=colors[2], linewidth=2.0)
plt.plot(t, results_turb2['F_aero_vec'][:, 0]/1000, label='TI = 0.0678', linestyle='--', color=colors[1], linewidth=2.0)
plt.plot(t, results_turb1['F_aero_vec'][:, 0]/1000, label='TI = 0.0493', linestyle='-', color=colors[0], linewidth=2.0, alpha = 0.8)
# Steady not plotted here, commented in your code

plt.xlabel('Time [s]', fontsize=16)
plt.ylabel(r'$F_{\mathrm{aero}}$ [kN]', fontsize=16)

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[3], handles[2], handles[1], handles[0]], [labels[3], labels[2], labels[1], labels[0]],
           loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'Fx_TI.pdf'), format='pdf')
plt.close()

# 4) Fy aerodynamic force vector plot
plt.figure(figsize=(10, 5))
plt.plot(t, results_turb4['F_aero_vec'][:, 1]/1000, label='TI = 0.1500', linestyle='-.', color=colors[3], linewidth=2.0)
plt.plot(t, results_turb3['F_aero_vec'][:, 1]/1000, label='TI = 0.0881', linestyle=':', color=colors[2], linewidth=2.0)
plt.plot(t, results_turb2['F_aero_vec'][:, 1]/1000, label='TI = 0.0678', linestyle='--', color=colors[1], linewidth=2.0)
plt.plot(t, results_turb1['F_aero_vec'][:, 1]/1000, label='TI = 0.0493', linestyle='-', color=colors[0], linewidth=2.0, alpha = 0.8)

plt.xlabel('Time [s]', fontsize=16)
plt.ylabel(r'$F_{\mathrm{aero}}$ [kN]', fontsize=16)

handles, labels = plt.gca().get_legend_handles_labels()
# Rearrange handles and labels to original legend order:
order = [3, 2, 1, 0]  # indices to reorder back to 9.5, 12.5, 15.5
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
           loc='upper center',
           bbox_to_anchor=(0.5, 1.15),
           ncol=4,
           fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'Fy_TI.pdf'), format='pdf')
plt.close()


plt.figure(figsize=(10, 5))
#plt.plot(t, results_noturb1['F_aero_vec'][:, 2], label='Steady', linestyle = '-', color = colors[0], linewidth=2.0)
plt.plot(t, results_turb4['F_aero_vec'][:, 2]/1000, label='TI = 0.1500', linestyle='-.', color=colors[3], linewidth=2.0)
plt.plot(t, results_turb3['F_aero_vec'][:, 2]/1000, label='TI = 0.0881', linestyle=':', color=colors[2], linewidth=2.0)
plt.plot(t, results_turb2['F_aero_vec'][:, 2]/1000, label='TI = 0.0678', linestyle='--', color=colors[1], linewidth=2.0)
plt.plot(t, results_turb1['F_aero_vec'][:, 2]/1000, label='TI = 0.0493', linestyle='-', color=colors[0], linewidth=2.0, alpha = 0.8)


plt.xlabel('Time [s]', fontsize=16)
plt.ylabel(r'$F_{\mathrm{aero}}$ [kN]', fontsize=16)
# manually create legend handles and labels:
handles, labels = plt.gca().get_legend_handles_labels()
# Rearrange handles and labels to original legend order:
order = [3, 2, 1, 0]  # indices to reorder back to 9.5, 12.5, 15.5
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
           loc='upper center',
           bbox_to_anchor=(0.5, 1.15),
           ncol=4,
           fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.ylim(0, -5000)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'Fz_TI.pdf'), format='pdf')
plt.close()


# --- Lift Comparison ---
plt.figure(figsize=(10, 5))
plt.style.use('tableau-colorblind10')
#plt.plot(t, results_noturb['lift'], label='Steady')
plt.plot(t, results_turb3['lift'], label='Unsteady', alpha=0.7)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Lift [N]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
#plt.savefig('lift_comparison_15.5.pdf')

# --- Drag Comparison ---
plt.figure(figsize=(10, 5))
plt.style.use('tableau-colorblind10')
#plt.plot(t, results_noturb['drag'], label='Steady')
plt.plot(t, results_turb3['drag'], label='Unsteady', alpha=0.7)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Drag [N]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
#plt.savefig('drag_comparison_15.5.pdf')

#Find lift drag ratio

#plt.savefig('lift_drag_ratio_comparison_9.5.png')


# --- Total Aerodynamic Force Comparison ---
plt.figure(figsize=(10, 5))
plt.style.use('tableau-colorblind10')
#plt.plot(t, results_noturb['F_aero'], label='Steady')
plt.plot(t, results_turb3['F_aero'], label='Unsteady', alpha=0.7)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel(r'$F_{\mathrm{aero}}$ [N]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('f_aero_mag_TI8.png')

# --- Aerodynamic Force Vectors in Body Frame Turbulence ---
plt.figure(figsize=(10, 5))
plt.plot(t, results_turb3['F_aero_vec'][:, 0], label='F_x (Body X)')
plt.plot(t, results_turb3['F_aero_vec'][:, 1], label='F_y (Body Y)')
plt.plot(t, results_turb3['F_aero_vec'][:, 2], label='F_z (Body Z)')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel(r'$F_{\mathrm{aero}}$ [N]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(10,5))

plt.loglog(smoothed_freqs_9_5_accel, smoothed_freqs_9_5_accel*smoothed_spectrum_9_5_accel, label='TI = 0.0493', linestyle = '-', color = colors[0], linewidth = 2.0)
plt.loglog(smoothed_freqs_12_5_accel, smoothed_freqs_9_5_accel*smoothed_spectrum_12_5_accel, label='TI = 0.0678', linestyle='--', color = colors[1], linewidth = 2.0)
plt.loglog(smoothed_freqs_15_5_accel, smoothed_freqs_9_5_accel*smoothed_spectrum_15_5_accel, label='TI = 0.0881', linestyle=':', color = colors[2], linewidth = 2.0)
plt.loglog(smoothed_freqs_TI_accel, smoothed_freqs_TI_accel*smoothed_spectrum_TI_accel, label='TI = 0.1500', linestyle='-.', color = colors[3], linewidth = 2.0)
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel(r'$fS(f)$ [$m^2/s^4$]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'smoothed_accel_TI.pdf'), format='pdf')
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(t [10:-10], spectral_accel_TI [10:-10], label='TI = 0.1500', color=colors[3],linestyle ='-.')
plt.plot(t [10:-10], spectral_accel_15_5 [10:-10], label='TI = 0.0881', color=colors[2],  linestyle =':')
plt.plot(t [10:-10], spectral_accel_12_5 [10:-10], label='TI = 0.0678', color=colors[1], linestyle ='--')
plt.plot(t [10:-10], spectral_accel_9_5 [10:-10], label='TI = 0.0493', color=colors[0], alpha=0.7,linestyle ='-')
plt.xlabel('Time [s]', fontsize = 16)
plt.ylabel(r'Acceleration [$m/s^2$]', fontsize = 16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
handles, labels = plt.gca().get_legend_handles_labels()
# Rearrange handles and labels to original legend order:
order = [3, 2, 1, 0]  # indices to reorder back to 9.5, 12.5, 15.5
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
           loc='upper center',
           bbox_to_anchor=(0.5, 1.15),
           ncol=4,
           fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'spectral_accel_time_TI.pdf'), format='pdf')
plt.close()

plt.figure(figsize=(10, 5))
plt.hist(spectral_accel_TI [10:-10], bins=40, color=colors[3], edgecolor='black', label='TI = 0.1500')
plt.hist(spectral_accel_15_5 [10:-10], bins=40, color=colors[2], edgecolor='black', label='TI = 0.0881', hatch = '+')
plt.hist(spectral_accel_12_5 [10:-10], bins=40, color=colors[1], alpha=0.8, edgecolor='black', label='TI = 0.0678', hatch = '//')
plt.hist(spectral_accel_9_5 [10:-10], bins=40, color=colors[0], alpha=0.6, edgecolor='black', label='TI = 0.0493', hatch ='')
plt.xlabel('P90 accelerations [m/s²]', fontsize = 16)
plt.ylabel('Count [-]', fontsize = 16)
order = [3, 2, 1, 0]  # indices to reorder back to 9.5, 12.5, 15.5
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
           loc='upper center',
           bbox_to_anchor=(0.5, 1.15),
           ncol=4,
           fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'accel_hist_TI.pdf'), format='pdf')
plt.close()

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x_gauss, gauss_pdf(x_gauss, mu_9_5, sigma_9_5), label='TI = 0.0493', color=colors[0], linestyle = '-')
plt.plot(x_gauss, gauss_pdf(x_gauss, mu_12_5, sigma_12_5), label='TI = 0.0678', color=colors[1], linestyle = '--')
plt.plot(x_gauss, gauss_pdf(x_gauss, mu_15_5, sigma_15_5), label='TI = 0.0881', color=colors[2], linestyle = ':')
plt.plot(x_gauss, gauss_pdf(x_gauss, mu_TI, sigma_TI), label='TI = 0.1500', color=colors[3], linestyle = '-.')
plt.xlabel('Accelerations [m/s²]', fontsize=16)
plt.ylabel(r'Probability Density [$m^{-1}s^2$]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'accel_gauss_pdfs_TI.pdf'), format='pdf')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(x_gauss,pdf_9_5, label='9.5 m/s', color=colors[0], linestyle = '-')
plt.plot(x_gauss, pdf_12_5, label='12.5 m/s', color=colors[1], linestyle = '--')
plt.plot(x_gauss, pdf_15_5, label='15.5 m/s', color=colors[2], linestyle = ':')

plt.xlabel('Accelerations [m/s²]', fontsize=16)
plt.ylabel('Probability Density [-]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'normalise-accel_gauss_pdfs.pdf'), format='pdf')
plt.close()
plt.show()

"""