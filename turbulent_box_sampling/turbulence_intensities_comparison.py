#Import packages
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hipersim import MannTurbulenceField
import pandas as pd
from scipy.signal import coherence
import os

#Import function files
from functions_spectrum import *
from turb_functions import *
from sampling_functions import *
from aerodynamics_functions import *

params_1 = {
        "U_mean": 15.5,
        "alpha_epsilon": 0.070903586,
        "L":129.7673497,
        "Gamma": 2.836620423,
        "desired_TI": 0.0493,
    }

params_2 =   {
        "U_mean":15.5,
        "alpha_epsilon": 0.0718741047377661,
        "L": 66.35046795800946		,
        "Gamma": 2.4268213212433256,
        "desired_TI": 0.0704,
    }
"""
params_3 =    {
        "U_mean": 15.5,
        "alpha_epsilon":  0.070903586,
        "L": 129.7673497	,
        "Gamma":2.836620423,
        "desired_TI": 0.0881,
    }
"""
sim_time = 300
dt = 0.1
#Umean = 12.5
# Generate the three boxes
data1_u, data1_v, data1_w, (DX1, DY1, DZ1) = generate_mann_box(
    params_1["U_mean"], params_1["desired_TI"], params_1["alpha_epsilon"], params_1["L"], params_1["Gamma"]
)
data2_u, data2_v, data2_w, (DX2, DY2, DZ2) = generate_mann_box(
    params_2["U_mean"], params_2["desired_TI"], params_2["alpha_epsilon"], params_2["L"], params_2["Gamma"]
)
#data3_u, data3_v, data3_w, (DX3, DY3, DZ3) = generate_mann_box(
   # params_3["U_mean"], params_3["desired_TI"], params_3["alpha_epsilon"], params_3["L"], params_3["Gamma"]
#)

# Now you have three turbulence boxes saved in variables data1_*, data2_*, data3_*
Nx, Ny, Nz = data1_u.shape
print("Box 1 shape:", data1_u.shape)
print("Box 2 shape:", data2_u.shape)
#print("Box 3 shape:", data3_u.shape)





# --- Sampling  of Turbulence Field ---
#Helical Sampling

# Helical sampling for each box
samples1, coords_grid1, coords_phys1 = helical_sample_velocity_field_physical_moving_box(
    data1_u, data1_v, data1_w, U_box=params_1["U_mean"],
    radius_m=60,
    T_loop=10,
    total_time=300,
    DX=DX1, DY=DY1, DZ=DZ1
)

samples2, coords_grid2, coords_phys2 = helical_sample_velocity_field_physical_moving_box(
    data2_u, data2_v, data2_w, U_box=params_2["U_mean"],
    radius_m=60,
    T_loop=10,
    total_time=300,
    DX=DX2, DY=DY2, DZ=DZ2
)
"""
samples3, coords_grid3, coords_phys3 = helical_sample_velocity_field_physical_moving_box(
    data3_u, data3_v, data3_w, U_box=params_3["U_mean"],
    radius_m=60,
    T_loop=10,
    total_time=300,
    DX=DX3, DY=DY3, DZ=DZ3
)

"""
# Assuming you have samples1, samples2, samples3
# and coords_phys1, coords_phys2, coords_phys3
# Also assuming U_mean1, U_mean2, U_mean3, Ny, Nz for each box

# You might want to extract Ny, Nz from each box's shape:
Ny1, Nz1 = data1_u.shape[1], data1_u.shape[2]
Ny2, Nz2 = data2_u.shape[1], data2_u.shape[2]
#Ny3, Nz3 = data3_u.shape[1], data3_u.shape[2]

# Define simulation time vector for each (assuming dt is global)
t1 = np.linspace(0, dt * len(samples1), len(samples1), endpoint=False)
t2 = np.linspace(0, dt * len(samples2), len(samples2), endpoint=False)
#t3 = np.linspace(0, dt * len(samples3), len(samples3), endpoint=False)

def run_aero_analysis(samples, coords_phys, U_mean, Ny, Nz, DY, DZ):
    x, y, z = coords_phys[:, 0], coords_phys[:, 1], coords_phys[:, 2]

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
        U_mean=U_mean, rho=1.225, A=2.982, CD0=0.004, a=0.008,
        DY=DY, DZ=DZ, Ny=Ny, Nz=Nz,
        fix_aoa_deg=6.0  
    )
    pitch_deg = results_noturb['awa_deg'] - results_noturb['aoa_deg']
    print(pitch_deg)
    # Run aero with turbulence
    results_turb = compute_aerodynamics(
        t1, x, y, z, v_global, samples,
        U_mean=U_mean, rho=1.225, A=2.982, CD0=0.004, a=0.008,
        DY=DY, DZ=DZ, Ny=Ny, Nz=Nz,
        pitch_reference=pitch_deg  
    )
    return results_turb, results_noturb

# Run for all three boxes
results_turb1, results_noturb1 = run_aero_analysis(samples1, coords_phys1, params_1["U_mean"], Ny1, Nz1, DY1, DZ1)
results_turb2, results_noturb2 = run_aero_analysis(samples2, coords_phys2, params_2["U_mean"], Ny2, Nz2, DY2, DZ2)
#results_turb3, results_noturb3 = run_aero_analysis(samples3, coords_phys3, params_3["U_mean"], Ny3, Nz3, DY3, DZ3)

# Define a list of results for each turbulent case
results_turb_list = [results_turb1, results_turb2]#, results_turb3]

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
Fx2, Fy2, Fz2 = results_turb2['F_aero_vec'][:, 0], results_turb2['F_aero_vec'][:, 1], results_turb2['F_aero_vec'][:, 2]
#Fx3, Fy3, Fz3 = results_turb3['F_aero_vec'][:, 0], results_turb3['F_aero_vec'][:, 1], results_turb3['F_aero_vec'][:, 2]

# Combine for looping
cases = [(Fx1, Fy1, Fz1, 'Unfiltered'),
         (Fx2, Fy2, Fz2, 'Filtered')]#,
         #(Fx3, Fy3, Fz3, 'U = 15.5 m/s')]





colors = ['#0072B2',  # Blue
          '#E69F00',  # Orange
          '#009E73', #Green
          '#CC79A7', # Pink
          '#D55E00',  # Vermillion
          '#F0E442']  # Yellow
# Get the directory where your script is
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the existing 'TI_comparison' folder
figures_dir = os.path.join(script_dir, 'Transients')
t = np.linspace(0, dt * len(samples1), len(samples1), endpoint=False)

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




plt.figure(figsize=(10, 5))
#plt.style.use('tableau-colorblind10')
plt.plot(t, results_turb1['airspeed'], label='Unfiltered', linestyle='-')   # solid
plt.plot(t, results_turb2['airspeed'], label='Filtered', linestyle='--')  # dashed
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

plt.plot(t, results_turb1['awa_deg'], label='Unfiltered', linestyle='-')
plt.plot(t, results_turb2['awa_deg'], label='Filtered', linestyle='--')
#plt.plot(t, results_turb3['awa_deg'], label='15.5 m/s', linestyle=':')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Apparent Wind Angle [deg]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()




plt.figure(figsize=(10, 5))
#plt.style.use('tableau-colorblind10')
#plt.plot(t, results_noturb1['aoa_deg'], label='Steady', linestyle='-', color = colors[0], linewidth=2.0)
plt.plot(t, results_turb1['aoa_deg'], label='Unfiltered', linestyle='-', color = colors[0], linewidth=2.0)
plt.plot(t, results_turb2['aoa_deg'], label='Filtered', linestyle='--', color = colors[1], linewidth=2.0)
#plt.plot(t, results_turb3['aoa_deg'], label='15.5 m/s', linestyle=':', color = colors[3], linewidth=2.0)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Angle of Attack [deg]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 16.5)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'aoa_transients_15_5.pdf'), format='pdf')
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



plt.figure(figsize=(10, 5))
ld_ratio_noturb1 = results_noturb1['lift'] / results_noturb1['drag']
ld_ratio_turb1 = results_turb1['lift'] / results_turb1['drag']
ld_ratio_noturb2 = results_noturb2['lift'] / results_noturb2['drag']
ld_ratio_turb2 = results_turb2['lift'] / results_turb2['drag']
#ld_ratio_turb3 = results_turb3['lift'] / results_turb3['drag']
#plt.plot(t, ld_ratio_noturb1, label='Steady', linestyle='-', color = colors[0], linewidth=2.0)  
plt.plot(t, ld_ratio_turb1, label='Unfiltered', alpha=0.7, linestyle='--', color = colors[0], linewidth=2.0)   
plt.plot(t, ld_ratio_turb2, label='Filtered', alpha=0.7, linestyle='--', color = colors[1], linewidth=2.0)  
#plt.plot(t, ld_ratio_turb3, label='15.5 m/s', alpha=0.7, linestyle='--', color = colors[3], linewidth=2.0)  
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('L/D [-]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(10, 90)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'LDratio_transients_15_5.pdf'), format='pdf')
plt.close()

# --- Aerodynamic Force Vectors in Body Frame Turbulence ---
plt.figure(figsize=(10, 5))
plt.plot(t, results_turb1['F_aero_vec'][:, 0], label='Unfiltered', linestyle = '-', color = colors[0], linewidth=2.0)
plt.plot(t, results_turb2['F_aero_vec'][:, 0], label='Filtered', linestyle ='--', color = colors[1], linewidth=2.0)
#plt.plot(t, results_turb3['F_aero_vec'][:, 0], label='15.5 m/s', linestyle = ':', color = colors[2], linewidth=2.0)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel(r'$F_{\mathrm{aero}}$ [N]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'Fx_transients_15_5.pdf'), format='pdf')
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(t, results_turb1['F_aero_vec'][:, 1], label='Unfiltered', linestyle = '-', color = colors[0], linewidth=2.0)
plt.plot(t, results_turb2['F_aero_vec'][:, 1], label='Filtered', linestyle ='--', color = colors[1], linewidth=2.0)
#plt.plot(t, results_turb3['F_aero_vec'][:, 1], label='15.5 m/s', linestyle = ':', color = colors[2], linewidth=2.0)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel(r'$F_{\mathrm{aero}}$ [N]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'Fy_transients_15_5.pdf'), format='pdf')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(t, results_turb1['F_aero_vec'][:, 2], label='Unfiltered', linestyle = '-', color = colors[0], linewidth=2.0)
plt.plot(t, results_turb2['F_aero_vec'][:, 2], label='Filtered', linestyle ='--', color = colors[1], linewidth=2.0)
#plt.plot(t, results_turb3['F_aero_vec'][:, 2], label='15.5 m/s', linestyle = ':', color = colors[2], linewidth=2.0)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel(r'$F_{\mathrm{aero}}$ [N]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, -5000)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'Fz_transients_15_5.pdf'), format='pdf')
plt.close()
plt.show()

"""
# --- Lift Comparison ---
plt.figure(figsize=(10, 5))
plt.style.use('tableau-colorblind10')
plt.plot(t, results_noturb['lift'], label='Steady')
plt.plot(t, results_turb['lift'], label='Unsteady', alpha=0.7)
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
plt.plot(t, results_noturb['drag'], label='Steady')
plt.plot(t, results_turb['drag'], label='Unsteady', alpha=0.7)
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
plt.plot(t, results_noturb['F_aero'], label='Steady')
plt.plot(t, results_turb['F_aero'], label='Unsteady', alpha=0.7)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel(r'$F_{\mathrm{aero}}$ [N]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
#plt.savefig('f_aero_comparison_9.5.png')

# --- Aerodynamic Force Vectors in Body Frame Turbulence ---
plt.figure(figsize=(10, 5))
plt.plot(t, results_turb['F_aero_vec'][:, 0], label='F_x (Body X)')
plt.plot(t, results_turb['F_aero_vec'][:, 1], label='F_y (Body Y)')
plt.plot(t, results_turb['F_aero_vec'][:, 2], label='F_z (Body Z)')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel(r'$F_{\mathrm{aero}}$ [N]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()

plt.show()

"""