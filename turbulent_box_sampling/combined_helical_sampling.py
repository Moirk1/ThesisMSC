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

"""velocity_field = np.stack((data_u, data_v, data_w), axis=0)
print("Combined Velocity Field Shape:", velocity_field.shape)"""

def helical_sample_velocity_field_physical_moving_box(data_u, data_v, data_w,
                                                      U_box, radius_m=60, T_loop=10, 
                                                      total_time=100, DX=0.3632, DY=7.5, DZ=7.5):
    """
    Sample a 3D turbulence field along a helical AWES path using physical parameters, accounting for box movement.

    Parameters:
    - U_box: Velocity of the turbulence box in the x-direction (in m/s)
    - radius_m: radius of helix in meters
    - T_loop: time (s) to complete one full loop
    - total_time: total sampling duration in seconds
    - DX, DY, DZ: grid resolution in x, y, z

    Returns:
    - helical_samples: array of shape (N, 3) with sampled [u, v, w]
    - sample_coords_grid: (N, 3) grid indices [x, y, z]
    - sample_coords_phys: (N, 3) physical coordinates [x, y, z]
    """
    Nx, Ny, Nz = data_u.shape

    # Kite's velocity relative to the box frame [Assumption that operating a optimal flight]
    U_kite_box = (1 / 3) * U_box

    # Set timestep for sampling
    dt = 0.1  # seconds per sample
    n_samples = int(total_time / dt)
    t = np.linspace(0, total_time, n_samples)
   
    # Helical path in physical space (inertial frame)
    x_phys = (U_box + U_kite_box) * t  # Total velocity of the kite in inertial frame
    y_phys = radius_m * np.cos(2 * np.pi * t / T_loop)
    z_phys = radius_m * np.sin(2 * np.pi * t / T_loop)

    # Center vertically in the turbulence box
    z_offset = (Nz * DZ) / 2
    y_offset = (Ny * DY) / 2

    y_phys += y_offset
    z_phys += z_offset

    # Convert to grid coordinates for sampling
    x_grid = x_phys / DX
    y_grid = y_phys / DY
    z_grid = z_phys / DZ
    sample_coords_grid = np.vstack((x_grid, y_grid, z_grid)).T
    sample_coords_phys = np.vstack((x_phys, y_phys, z_phys)).T

    # Interpolate velocity values
    u_vals = map_coordinates(data_u, sample_coords_grid.T, order=1, mode='nearest')
    v_vals = map_coordinates(data_v, sample_coords_grid.T, order=1, mode='nearest')
    w_vals = map_coordinates(data_w, sample_coords_grid.T, order=1, mode='nearest')
    helical_samples = np.vstack((u_vals, v_vals, w_vals)).T
    print("Helical samples:", helical_samples.shape)
    return helical_samples, sample_coords_grid, sample_coords_phys


# --- Sampling and visualization parameters ---

samples, coords_grid, coords_phys = helical_sample_velocity_field_physical_moving_box(
    data_u, data_v, data_w, U_box = 8.5,
    radius_m=60,
    T_loop=11,
    total_time=330,
    DX=0.3632, DY=7.5, DZ=7.5
)

# --- Plot Turbulence components ---
plt.figure(figsize=(10, 5))
dt = 0.1  # seconds per sample
t = np.linspace(0, dt * len(samples), len(samples), endpoint=False)

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


# --- Plot helical path in 3D with physical units (meters) ---
x, y, z = coords_phys[:, 0], coords_phys[:, 1], coords_phys[:, 2]
DX = 0.3632
DY = 7.5
DZ = 7.5
box_x = Nx * DX
box_y = Ny * DY
box_z = Nz * DZ

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, color='purple', linewidth=2)
ax.scatter(x[0], y[0], z[0], color='green', s=60, label='Start')
ax.scatter(x[-1], y[-1], z[-1], color='red', s=60, label='End')

ax.set_xlim(0, box_x)
ax.set_ylim(0, box_y)
ax.set_zlim(0, box_z)

# Set label fonts
ax.set_xlabel('X [m]', fontsize=16, labelpad=12)
ax.set_ylabel('Y [m]', fontsize=16, labelpad=12)
ax.set_zlabel('Z [m]', fontsize=16, labelpad=12)

# Set tick label fonts
ax.tick_params(axis='both', which='major', labelsize=14)
for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
    label.set_fontsize(14)

# Grid customization
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
plt.show()

