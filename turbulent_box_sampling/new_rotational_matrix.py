#Import packages
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hipersim import MannTurbulenceField
import pandas as pd
from scipy.signal import coherence

#Import function files
from functions_spectrum import *
from turb_functions import *
from sampling_functions import *
from aerodynamics_functions import *

#Define Mann Parameters & Simulation time
U_mean = 12.5 #CHANGE SAVE FIG NAME
alpha_epsilon = 0.0456825588131349
L = 68.05627895300304		
Gamma = 2.438897937014391
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

# Extract components from helical sampling
u_h = samples[:, 0]
v_h = samples[:, 1]
w_h = samples[:, 2]

# Extract components from centreline sampling
u_l = samples_linear[:, 0]
v_l = samples_linear[:, 1]
w_l = samples_linear[:, 2]

# Calculate mean and standard deviation for each
def describe_component(name, data):
    mean = np.mean(data)
    std = np.std(data)
    print(f"{name}: Mean = {mean:.4f}, Std Dev = {std:.4f}")
    return mean, std

print("Helical Sampling:")
describe_component("u_h", u_h)
describe_component("v_h", v_h)
describe_component("w_h", w_h)

print("\nCentreline Sampling:")
describe_component("u_l", u_l)
describe_component("v_l", v_l)
describe_component("w_l", w_l)

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

#Extracting u,v,w for plotting
freqs_h_u, psd_h_u = psd_helical['u']
freqs_h_v, psd_h_v = psd_helical['v']
freqs_h_w, psd_h_w = psd_helical['w']

freqs_l_u, psd_l_u = psd_linear['u']
freqs_l_v, psd_l_v = psd_linear['v']
freqs_l_w, psd_l_w = psd_linear['w']

speccheck_unsmooth("Helical_sampling", samples[:,0], freqs_h_u, psd_h_u)
speccheck_unsmooth("Linear_sampling", samples_linear[:,0], freqs_l_u, psd_l_u)

# Smoothing
n_per_decade = 20  # Set to a value between ~10 and 20 [Need to look at how much I want to smooth]
smoothed_freqs_helical_u, smoothed_spectrum_helical_u = smooth_criminal(psd_h_u, freqs_h_u, n_per_decade)
smoothed_freqs_linear_u, smoothed_spectrum_linear_u = smooth_criminal(psd_l_u, freqs_l_u, n_per_decade)
smoothed_freqs_helical_v, smoothed_spectrum_helical_v = smooth_criminal(psd_h_v, freqs_h_v, n_per_decade)
smoothed_freqs_linear_v, smoothed_spectrum_linear_v = smooth_criminal(psd_l_v, freqs_l_v, n_per_decade)
smoothed_freqs_helical_w, smoothed_spectrum_helical_w = smooth_criminal(psd_h_w, freqs_h_w, n_per_decade)
smoothed_freqs_linear_w, smoothed_spectrum_linear_w = smooth_criminal(psd_l_w, freqs_l_w, n_per_decade)


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

#Find number of exceedances
events = find_aoa_exceedances(results_turb['aoa_deg'], dt, threshold = 10, min_duration = 1)

for idx, (start, end) in enumerate(events):
    print(f"Event {idx+1}: AoA > 10 degrees from {start:.2f}s to {end:.2f}s (duration {end-start:.2f}s)")

# --- Plotting ---
nperseg = 256
f_u, coh_u = coherence(u_h, u_l, fs=fs, nperseg=nperseg)
f_v, coh_v = coherence(v_h, u_l, fs=fs, nperseg=nperseg)
f_w, coh_w = coherence(w_h, w_l, fs=fs, nperseg=nperseg)

plt.figure(figsize=(10, 5))
plt.style.use('tableau-colorblind10')
plt.semilogx(f_u, coh_u, label='u')
plt.semilogx(f_v, coh_v, label='v')
plt.semilogx(f_w, coh_w, label='w')
plt.xlabel('Frequency [Hz]', fontsize = 16)
plt.ylabel('Coherence', fontsize = 16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
#plt.savefig('coherences_sampling_method.pdf')


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
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
#plt.savefig('turbulence_experienced_helical.pdf')


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
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
#plt.savefig('turbulence_experienced_linear.pdf')
"""
# Choose the slice at y=0
y_index = 0

# Get the 2D slice: u(x, z) at y=0
u_slice = data_u[:, y_index, :]  # shape: (Nx, Nz)

# Calculate physical axes in meters
DX = 0.3632  # your grid spacing in x (meters)
DZ = 7.5     # your grid spacing in z (meters)
x_meters = np.arange(u_slice.shape[0]) * DX
z_meters = np.arange(u_slice.shape[1]) * DZ
# Plot the heatmap with red-white-blue colormap
plt.figure(figsize=(10, 5))
plt.imshow(
    u_slice.T,
    aspect='auto',
    origin='lower',
    cmap='bwr',
    extent=[x_meters[0], x_meters[-1], z_meters[0], z_meters[-1]],
    interpolation='nearest'
)
cbar = plt.colorbar()
cbar.set_label('u [m/s]', fontsize=14)
plt.xlabel('x [m]', fontsize=14)
plt.ylabel('z [m]', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("y = 0, uvw = 'u'", fontsize=16)
plt.grid(False)
plt.tight_layout()
plt.savefig('turbulence_box_slice.pdf')"""

#PSDS
"""plt.figure(figsize=(10,5))
plt.loglog(smoothed_freqs_helical_u, smoothed_freqs_helical_u*smoothed_spectrum_helical_u, label='Helical')
plt.loglog(smoothed_freqs_linear_u, smoothed_freqs_linear_u*smoothed_spectrum_linear_u, label='Centreline', linestyle='--')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel(r'$fS(f)$ [m$^2$/s$^2$]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig('psd_u_helvslin.pdf')

plt.figure(figsize=(10,5))

plt.loglog(smoothed_freqs_helical_v, smoothed_freqs_helical_v*smoothed_spectrum_helical_v, label='Helical')
plt.loglog(smoothed_freqs_linear_v, smoothed_freqs_linear_v*smoothed_spectrum_linear_v, label='Centreline', linestyle='--')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel(r'$fS(f)$ [m$^2$/s$^2$]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig('psd_v_helvslin.pdf')

plt.figure(figsize=(10,5))
plt.loglog(smoothed_freqs_helical_w, smoothed_freqs_helical_w*smoothed_spectrum_helical_w, label='Helical')
plt.loglog(smoothed_freqs_linear_w, smoothed_freqs_linear_w*smoothed_spectrum_linear_w, label='Centreline', linestyle='--')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel(r'$fS(f)$ [m$^2$/s$^2$]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig('psd_w_helvslin.pdf')


plt.figure(figsize=(10,5))
plt.semilogx(smoothed_freqs_helical_u, smoothed_freqs_helical_u*smoothed_spectrum_helical_u, label='Helical')
plt.semilogx(smoothed_freqs_linear_u, smoothed_freqs_linear_u*smoothed_spectrum_linear_u, label='Linear', linestyle='--')
plt.axvline(x=0.1, color='red', linestyle='--', linewidth=2, alpha=0.7)
plt.ylabel(r'$fS(f)$ [m$^2$/s$^2$]', fontsize=16)
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig('psdsemilog_u_helvslin.pdf')


plt.figure(figsize=(10,5))
plt.semilogx(smoothed_freqs_helical_v, smoothed_freqs_helical_v*smoothed_spectrum_helical_v, label='Helical')
plt.semilogx(smoothed_freqs_linear_v, smoothed_freqs_linear_v*smoothed_spectrum_linear_v, label='Linear', linestyle='--')
plt.axvline(x=0.1, color='red', linestyle='--', linewidth=2, alpha=0.7)
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel(r'$fS(f)$ [m$^2$/s$^2$]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig('psdsemilog_v_helvslin.pdf')



plt.figure(figsize=(10,5))
plt.semilogx(smoothed_freqs_helical_w, smoothed_freqs_helical_w*smoothed_spectrum_helical_w, label='Helical')
plt.semilogx(smoothed_freqs_linear_w, smoothed_freqs_linear_w*smoothed_spectrum_linear_w, label='Linear', linestyle='--')
plt.axvline(x=0.1, color='red', linestyle='--', linewidth=2, alpha=0.7)
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel(r'$fS(f)$ [m$^2$/s$^2$]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7, color='gray')
plt.tight_layout()
plt.savefig('psdsemilog_w_helvslin.pdf')


#Visualising Helix in Box
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, color='grey', linewidth=2)
ax.scatter(x[0], y[0], z[0], color='blue', s=60, label='Start')
ax.scatter(x[-1], y[-1], z[-1], color='orange', s=60, label='End')
box_x = Nx * 0.3632
box_y = Ny * 7.5
box_z = Nz * 7.5
ax.set_xlim(0, box_x)  # Limit x-axis to 2000 meters)
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
#plt.savefig('aoa_9.5_comparison.png')


plt.figure(figsize=(10,5))
plt.plot(t, results_turb['aoa_deg'], label="Angle of Attack (deg)")
plt.axhline(10, color='red', linestyle='--', label="10 degree threshold")

for (start, end) in events:
    plt.axvspan(start, end, color='orange', alpha=0.3)

plt.xlabel("Time (s)")
plt.ylabel("Angle of Attack (deg)")
plt.legend()



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
plt.tight_layout()"""

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

# --- Lift Comparison ---
plt.figure(figsize=(10, 5))
plt.style.use('tableau-colorblind10')
plt.plot(t, results_turb['drag_vec'][:, 0], label='D_x (Body X)')
plt.plot(t, results_turb['drag_vec'][:, 1], label='D_y (Body Y)')
plt.plot(t, results_turb['drag_vec'][:, 2], label='D_z (Body Z)')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Drag [N]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
#plt.savefig('drag_comparison_15.5.pdf')

# --- Drag Comparison ---
plt.figure(figsize=(10, 5))
plt.style.use('tableau-colorblind10')
plt.plot(t, results_turb['lift_vec'][:, 0], label='L_x (Body X)')
plt.plot(t, results_turb['lift_vec'][:, 1], label='L_y (Body Y)')
plt.plot(t, results_turb['lift_vec'][:, 2], label='L_z (Body Z)')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Lift [N]', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
#plt.savefig('drag_comparison_15.5.pdf')

plt.figure(figsize=(10, 5))
plt.style.use('tableau-colorblind10')
plt.plot(t, results_turb['v_apparent_vec'][:, 0], label='V_a (Body X)')
plt.plot(t, results_turb['v_apparent_vec'][:, 1], label='V_a (Body Y)')
plt.plot(t, results_turb['v_apparent_vec'][:, 2], label='V_a (Body Z)')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('V_apparent [m/s]', fontsize=16)
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
#plt.savefig('aoa_9.5_comparison.png')

plt.show()

