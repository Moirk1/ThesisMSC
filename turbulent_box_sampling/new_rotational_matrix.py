import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Load turbulence data ---
file_u = r"C:\Users\markj\OneDrive - KTH\DTU 2024-2025\Master thesis\Code\turbulent_box_generation_example\constrained_box_shearShifted_wind_speed_8.5_event_200907211500_seed_1000_1000_u.bin.ref"
file_v = r"C:\Users\markj\OneDrive - KTH\DTU 2024-2025\Master thesis\Code\turbulent_box_generation_example\constrained_box_shearShifted_wind_speed_8.5_event_200907211500_seed_1000_1000_v.bin.ref"
file_w = r"C:\Users\markj\OneDrive - KTH\DTU 2024-2025\Master thesis\Code\turbulent_box_generation_example\constrained_box_shearShifted_wind_speed_8.5_event_200907211500_seed_1000_1000_w.bin.ref"

data_u = np.fromfile(file_u, dtype=np.float32).reshape((16384, 32, 32))
data_v = np.fromfile(file_v, dtype=np.float32).reshape((16384, 32, 32))
data_w = np.fromfile(file_w, dtype=np.float32).reshape((16384, 32, 32))

Nx, Ny, Nz = data_u.shape
print("Data Shape (u, v, w):", data_u.shape, data_v.shape, data_w.shape)

# --- FUNCTIONS ---

# Samples a turbulent velocity field along a helical trajectory through a moving wind box
def helical_sample_velocity_field_physical_moving_box(data_u, data_v, data_w,
                                                      U_box, radius_m=60, T_loop=10, 
                                                      total_time=100, DX=0.3632, DY=7.5, DZ=7.5):

    Nx, Ny, Nz = data_u.shape
    U_kite_box = 1/3 * U_box
    dt = 0.1  # seconds per sample
    n_samples = int(total_time / dt)
    t = np.linspace(0, total_time, n_samples)
   
    x_phys = (U_box + U_kite_box) * t
    y_phys = radius_m * np.cos(2 * np.pi * t / T_loop)
    z_phys = radius_m * np.sin(2 * np.pi * t / T_loop)

    z_offset = (Nz * DZ) / 2
    y_offset = (Ny * DY) / 2
    y_phys += y_offset
    z_phys += z_offset

    x_grid = x_phys / DX
    y_grid = y_phys / DY
    z_grid = z_phys / DZ
    sample_coords_grid = np.vstack((x_grid, y_grid, z_grid)).T
    sample_coords_phys = np.vstack((x_phys, y_phys, z_phys)).T

    u_vals = map_coordinates(data_u, sample_coords_grid.T, order=1, mode='nearest')
    v_vals = map_coordinates(data_v, sample_coords_grid.T, order=1, mode='nearest')
    w_vals = map_coordinates(data_w, sample_coords_grid.T, order=1, mode='nearest')
    helical_samples = np.vstack((u_vals, v_vals, w_vals)).T
    print("Helical samples:", helical_samples.shape)
    return helical_samples, sample_coords_grid, sample_coords_phys

#Computes aerodynamic forces and body-frame velocities of a kite moving through a turbulent wind field.
def compute_aerodynamics(t, x, y, z, v_global, samples,
                         U_mean, rho, A, CD0, a, DY, DZ, Ny, Nz,
                         fix_aoa_deg=None, pitch_reference=None):
    airspeeds = []
    apparent_wind_angle_deg = []
    lift = []
    drag = []
    F_aero = []
    aoa_deg_corrected = []

    v_body_x = []
    v_body_y = []
    v_body_z = []

    y_offset = (Ny * DY) / 2
    z_offset = (Nz * DZ) / 2

    for i in range(len(t)):
        v_kite_global_i = v_global[i]
        v_windglobal_i = np.array([U_mean, 0, 0]) + samples[i, :]

        # Body-frame basis
        X_body = v_kite_global_i / np.linalg.norm(v_kite_global_i)
        radial_vec = np.array([0, y_offset - y[i], z_offset - z[i]])
        Y_body = -radial_vec / np.linalg.norm(radial_vec)
        Z_body = np.cross(X_body, Y_body)
        Z_body /= np.linalg.norm(Z_body)
        Y_body = np.cross(Z_body, X_body)
        Y_body /= np.linalg.norm(Y_body)

        R_nb = np.column_stack((X_body, Y_body, Z_body))

        v_kite_body = R_nb.T @ v_kite_global_i
        v_wind_body = R_nb.T @ v_windglobal_i
        v_apparent = v_kite_body - v_wind_body

        airspeed = np.linalg.norm(v_apparent)
        awa_rad = np.arctan2(v_apparent[2], v_apparent[0])  # AoA
        awa_deg = np.degrees(awa_rad)

        # Default AoA = AWA if nothing fixed
        aoa_deg = awa_deg
        if fix_aoa_deg is not None and pitch_reference is None:
            # We're computing pitch from fixed AoA
            aoa_deg = fix_aoa_deg
        elif pitch_reference is not None:
            # We're using provided pitch to compute AoA
            aoa_deg = awa_deg - pitch_reference[i]

        aoa_rad = np.radians(aoa_deg)

        # Aerodynamics
        C_L = 2 * np.pi * aoa_rad
        C_D_airfoil = CD0 + a * C_L**2
        q = 0.5 * rho * airspeed**2

        L = q * A * C_L
        D = q * A * C_D_airfoil
        F_a = np.sqrt(L**2 + D**2)

        # Store
        airspeeds.append(airspeed)
        apparent_wind_angle_deg.append(awa_deg)
        aoa_deg_corrected.append(aoa_deg)
        lift.append(L)
        drag.append(D)
        F_aero.append(F_a)
        v_body_x.append(v_kite_body[0])
        v_body_y.append(v_kite_body[1])
        v_body_z.append(v_kite_body[2])

    return {
        'airspeed': np.array(airspeeds),
        'awa_deg': np.array(apparent_wind_angle_deg),
        'aoa_deg': np.array(aoa_deg_corrected),
        'lift': np.array(lift),
        'drag': np.array(drag),
        'F_aero': np.array(F_aero),
        'v_body': np.vstack((v_body_x, v_body_y, v_body_z)).T
    }



# --- Sampling  of Turbulence Field ---
samples, coords_grid, coords_phys = helical_sample_velocity_field_physical_moving_box(
    data_u, data_v, data_w, U_box = 8.5,
    radius_m=60,
    T_loop=10,
    total_time=330,
    DX=0.3632, DY=7.5, DZ=7.5
)

samples_noturb = np.zeros_like(samples)

dt = 0.1
t = np.linspace(0, dt * len(samples), len(samples), endpoint=False)
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

#Compute the circumferential angle
DY = 7.5
DZ = 7.5
y_offset = (Ny * DY) / 2
z_offset = (Nz * DZ) / 2
phi_rad = np.arctan2(z - z_offset, y - y_offset)
phi_deg = np.degrees(phi_rad)

#---Calculate Aero parameters for steady & unsteady wind---
results_noturb = compute_aerodynamics(
    t, x, y, z, v_global, samples_noturb,
    U_mean=8.5, rho=1.225, A=2.982, CD0=0.004, a=0.008,
    DY=7.5, DZ=7.5, Ny=Ny, Nz=Nz,
    fix_aoa_deg=6.0  
)

pitch_deg = results_noturb['awa_deg'] - results_noturb['aoa_deg']
print(pitch_deg)

results_turb = compute_aerodynamics(
    t, x, y, z, v_global, samples,
    U_mean=8.5, rho=1.225, A=2.982, CD0=0.004, a=0.008,
    DY=7.5, DZ=7.5, Ny=Ny, Nz=Nz,
    pitch_reference=pitch_deg  
)


# --- Plotting ---

#Turbulence Sampled
plt.figure(figsize=(10, 5))
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
plt.savefig('turbulence_experienced.pdf')

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
plt.savefig('Helical_path_physical_units.pdf')

#Circumferential Angle
plt.figure(figsize=(10, 5))
plt.plot(t, phi_deg, label='Circumferential angle φ [deg]')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Angle [deg]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('angles_gamma_phi.pdf')

#Airspeed
plt.figure(figsize=(10, 5))
plt.plot(t, results_noturb['airspeed'], label='Airspeed without turbulence', color='blue')
plt.plot(t, results_turb['airspeed'], label='Airspeed with turbulence', color='orange', alpha=0.7)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Airspeed [m/s]', fontsize=16)
plt.title('Airspeed Comparison', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()

#Apparent Wind Angle
plt.figure(figsize=(10, 5))
plt.plot(t, results_noturb['awa_deg'], label='AWA without turbulence', color='green')
plt.plot(t, results_turb['awa_deg'], label='AWA with turbulence', color='red', alpha=0.7)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Apparent Wind Angle [deg]', fontsize=16)
plt.title('Apparent Wind Angle Comparison', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()


# AoA Comparison Plot
plt.figure(figsize=(10, 5))
plt.plot(t, results_noturb['aoa_deg'], label='AoA without turbulence', color='purple')
plt.plot(t, results_turb['aoa_deg'], label='AoA with turbulence', color='brown', alpha=0.7)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Angle of Attack [deg]', fontsize=16)
plt.title('Angle of Attack (AoA) Comparison', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()



# Plot Helix Frame velocity components
plt.figure(figsize=(10, 5))
plt.plot(t, dx, label='vx [m/s]')
plt.plot(t, dy, label='vy [m/s]')
plt.plot(t, dz, label='vz [m/s]')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Velocity [m/s]', fontsize=16)
plt.title("Kite Velocity Components in Global Frame", fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()

# --- Lift Comparison ---
plt.figure(figsize=(10, 5))
plt.plot(t, results_noturb['lift'], label='Lift without turbulence', color='blue')
plt.plot(t, results_turb['lift'], label='Lift with turbulence', color='orange', alpha=0.7)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Lift [N]', fontsize=16)
plt.title('Lift Force Comparison', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
#plt.savefig('lift_comparison.pdf')

# --- Drag Comparison ---
plt.figure(figsize=(10, 5))
plt.plot(t, results_noturb['drag'], label='Drag without turbulence', color='blue')
plt.plot(t, results_turb['drag'], label='Drag with turbulence', color='orange', alpha=0.7)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Drag [N]', fontsize=16)
plt.title('Drag Force Comparison', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
#plt.savefig('drag_comparison.pdf')

# --- Total Aerodynamic Force Comparison ---
plt.figure(figsize=(10, 5))
plt.plot(t, results_noturb['F_aero'], label='F_aero without turbulence', color='blue')
plt.plot(t, results_turb['F_aero'], label='F_aero with turbulence', color='orange', alpha=0.7)
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('F_aero [N]', fontsize=16)
plt.title('Total Aerodynamic Force Comparison', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
#plt.savefig('f_aero_comparison.pdf')

plt.show()

