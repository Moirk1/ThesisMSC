#Import packages
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hipersim import MannTurbulenceField
import pandas as pd

#Import function files
from functions_spectrum import *
from turb_functions import *
from sampling_functions import *
from aerodynamics_functions import *

#Define Mann Parameters & Simulation time
U_mean = 9.5 #CHANGE SAVE FIG NAME
alpha_epsilon = 0.023132626456368
L = 37.38544456749976
Gamma = 3.8408936134736287
desired_TI = 0.1
sim_time = 300
dt = 0.1

#Buckle your seatbelts, we're in for a bumpy ride!!!
#Turbulence options

# --- Load turbulence data ---
#file_u = r"C:\Users\markj\OneDrive - KTH\DTU 2024-2025\Master thesis\Code\turbulent_box_generation_example\constrained_box_shearShifted_wind_speed_8.5_event_200907211500_seed_1000_1000_u.bin.ref"
#file_v = r"C:\Users\markj\OneDrive - KTH\DTU 2024-2025\Master thesis\Code\turbulent_box_generation_example\constrained_box_shearShifted_wind_speed_8.5_event_200907211500_seed_1000_1000_v.bin.ref"
#file_w = r"C:\Users\markj\OneDrive - KTH\DTU 2024-2025\Master thesis\Code\turbulent_box_generation_example\constrained_box_shearShifted_wind_speed_8.5_event_200907211500_seed_1000_1000_w.bin.ref"
#data_u = np.fromfile(file_u, dtype=np.float32).reshape((16384, 32, 32))
#data_v = np.fromfile(file_v, dtype=np.float32).reshape((16384, 32, 32))
#data_w = np.fromfile(file_w, dtype=np.float32).reshape((16384, 32, 32))
#Nx, Ny, Nz = data_u.shape
#print("Data Shape (u, v, w):", data_u.shape, data_v.shape, data_w.shape)


# --- Run everything --- [FOR RUNNING ALL THE TURB BOXES] [CAREFUL TAKES A LONG TIME]
#if __name__ == "__main__":
    #folder_path = r"C:\Users\markj\OneDrive - KTH\DTU 2024-2025\Master thesis\Code\turbulent_boxes\wind_data_from_Mark\Timeseries_filtered_O2butter_0.1Hz\Hov_au160max_MMparams"
    #output_summary_file = os.path.join(folder_path, "mann_TI_summary.txt")
    #process_files_in_directory(folder_path, output_summary_file) 


# --- Generate Mann box ---
data_u, data_v, data_w, (DX, DY, DZ) = generate_mann_box(U_mean, desired_TI, alpha_epsilon, L, Gamma )
Nx, Ny, Nz = data_u.shape
print("Data Shape (u, v, w):", data_u.shape, data_v.shape, data_w.shape)



# --- Sampling  of Turbulence Field ---
#Helical Sampling
samples, coords_grid, coords_phys = helical_sample_velocity_field_physical_moving_box(
    data_u, data_v, data_w, U_box = U_mean,
    radius_m=60,
    T_loop=10,
    total_time=300,
    DX=0.3632, DY=7.5, DZ=7.5
)
#Linear Sampling
samples_linear, coords_grid_linear, coords_phys_linear = linear_sample_velocity_field(data_u, data_v, data_w, U_box = U_mean,
    total_time=300,
    DX=0.3632, DY=7.5, DZ=7.5)

#Create an array for no turbulence Case
samples_noturb = np.zeros_like(samples)

#Define length of simulation
t = np.linspace(0, dt * len(samples), len(samples), endpoint=False)

#Find x,y,z 
x, y, z = coords_phys[:, 0], coords_phys[:, 1], coords_phys[:, 2]

# --- Compute velocity and acceleration ---
dx = np.gradient(x, dt)
dy = np.gradient(y, dt)
dz = np.gradient(z, dt)
v_global = np.vstack((dx, dy, dz)).T
v_mag = np.sqrt(dx**2 + dy**2 + dz**2)

dvx = np.gradient(dx, dt)
dvy = np.gradient(dy, dt)
dvz = np.gradient(dz, dt)
a_global = np.vstack((dvx, dvy, dvz)).T

#Compute the circumferential angle (know where in helix kite is)
DY = 7.5
DZ = 7.5
y_offset = (Ny * DY) / 2
z_offset = (Nz * DZ) / 2
phi_rad = np.arctan2(z - z_offset, y - y_offset)
phi_deg = np.degrees(phi_rad)

#---Spectral stuff ----
fs = 1 / dt  # Sampling frequency = 10 Hz
#For helical
psd_helical = compute_psd_all_components(samples, fs)
# For linear sampling
psd_linear = compute_psd_all_components(samples_linear, fs)
freqs_h, psd_h = psd_helical['u']
freqs_l, psd_l = psd_linear['u']
speccheck_unsmooth("Helical_sampling", samples[:,0], freqs_h, psd_h)
speccheck_unsmooth("Linear_sampling", samples_linear[:,0], freqs_l, psd_l)

# Smoothing
n_per_decade = 20  # Set to a value between ~10 and 20 [Need to look at how much I want to smooth]
smoothed_freqs_helical, smoothed_spectrum_helical = smooth_criminal(psd_h, freqs_h, n_per_decade)
smoothed_freqs_linear, smoothed_spectrum_linear = smooth_criminal(psd_l, freqs_l, n_per_decade)



#---Calculate Aero parameters for steady & unsteady wind---
results_noturb = compute_aerodynamics(
    t, x, y, z, v_global, samples_noturb,
    U_mean, rho=1.225, A=2.982, CD0=0.004, a=0.008,
    DY=7.5, DZ=7.5, Ny=Ny, Nz=Nz,
    fix_aoa_deg=6.0  
)

pitch_deg = results_noturb['awa_deg'] - results_noturb['aoa_deg']
print(pitch_deg)

results_turb = compute_aerodynamics(
    t, x, y, z, v_global, samples,
    U_mean, rho=1.225, A=2.982, CD0=0.004, a=0.008,
    DY=7.5, DZ=7.5, Ny=Ny, Nz=Nz,
    pitch_reference=pitch_deg  
)

# --- Plotting ---

#Turbulence Sampled Helix
plt.figure(figsize=(10, 5))
plt.style.use('tableau-colorblind10')
plt.plot(t, samples[:, 0], label='u')
plt.plot(t, samples[:, 1], label='v')
plt.plot(t, samples[:, 2], label='w')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Velocity [m/s]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()

#Turbulence Sampled linear
plt.figure(figsize=(10, 5))
plt.style.use('tableau-colorblind10')
plt.plot(t, samples_linear[:, 0], label='u')
plt.plot(t, samples_linear[:, 1], label='v')
plt.plot(t, samples_linear[:, 2], label='w')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Velocity [m/s]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.title('linear')
plt.tight_layout()
#plt.savefig('turbulence_experienced.pdf')

#PSD
plt.figure(figsize=(10,5))
plt.loglog(freqs_h, freqs_h*psd_h, label='Helical u')
plt.loglog(freqs_l, freqs_l*psd_l, label='Linear u')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectral Density')
plt.legend()

plt.figure(figsize=(10,5))
plt.loglog(smoothed_freqs_helical, smoothed_freqs_helical*smoothed_spectrum_helical, label='Helical u')
plt.loglog(smoothed_freqs_linear, smoothed_freqs_linear*smoothed_spectrum_linear, label='Linear u')
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'fS(f) [$m^2/s^2$]')
plt.legend()



#Visualising Helix in Box
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, color='purple', linewidth=2)
ax.scatter(x[0], y[0], z[0], color='green', s=60, label='Start')
ax.scatter(x[-1], y[-1], z[-1], color='red', s=60, label='End')
box_x = Nx * 0.3632
box_y = Ny * 7.5
box_z = Nz * 7.5
ax.set_xlim(0, box_x)
ax.set_ylim(0, box_y)
ax.set_zlim(0, box_z)
ax.set_xlabel('X [m]', fontsize=16)
ax.set_ylabel('Y [m]', fontsize=16)
ax.set_zlabel('Z [m]', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(True)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis._axinfo['grid'].update(color='lightgray', linestyle='--')
ax.yaxis._axinfo['grid'].update(color='lightgray', linestyle='--')
ax.zaxis._axinfo['grid'].update(color='lightgray', linestyle='--')
ax.legend(fontsize=16)
plt.tight_layout()
#plt.savefig('Helical_path_physical_units.pdf')

#Circumferential Angle
plt.figure(figsize=(10, 5))
plt.style.use('tableau-colorblind10')
plt.plot(t, phi_deg, label='Circumferential angle φ [deg]')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Angle [deg]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
#plt.savefig('angles_gamma_phi.pdf')

#Airspeed
plt.figure(figsize=(10, 5))
plt.style.use('tableau-colorblind10')
plt.plot(t, results_noturb['airspeed'], label='Steady')
plt.plot(t, results_turb['airspeed'], label='Unsteady', alpha=0.7)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Airspeed [m/s]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()

#Apparent Wind Angle
plt.figure(figsize=(10, 5))
plt.style.use('tableau-colorblind10')
plt.plot(t, results_noturb['awa_deg'], label='Steady')
plt.plot(t, results_turb['awa_deg'], label='Unsteady', alpha=0.7)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Apparent Wind Angle [deg]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()


# AoA Comparison Plot
plt.figure(figsize=(10, 5))
plt.style.use('tableau-colorblind10')
plt.plot(t, results_noturb['aoa_deg'], label='Steady')
plt.plot(t, results_turb['aoa_deg'], label='Unsteady', alpha=0.7)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Angle of Attack [deg]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('aoa_9.5_comparison.png')


# Plot Helix Frame velocity components
plt.figure(figsize=(10, 5))
plt.style.use('tableau-colorblind10')
plt.plot(t, dx, label='vx [m/s]')
plt.plot(t, dy, label='vy [m/s]')
plt.plot(t, dz, label='vz [m/s]')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Velocity [m/s]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()

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
plt.figure(figsize=(10, 5))
plt.style.use('tableau-colorblind10')
ld_ratio_noturb = results_noturb['lift'] / results_noturb['drag']
ld_ratio_turb = results_turb['lift'] / results_turb['drag']
plt.plot(t, ld_ratio_noturb, label='Steady')  # Nice blue
plt.plot(t, ld_ratio_turb, label='Unsteady', alpha=0.7)  # Light pink
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('L/D [-]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('lift_drag_ratio_comparison_9.5.png')


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

