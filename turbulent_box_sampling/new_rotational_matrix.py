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

"""
def calculate_airspeed_and_aoa_from_global(v_kite_global, v_windglobal, R_total):
    v_wind_body = R_total @ v_windglobal
    v_kite_body = R_total @ v_kite_global
    v_apparent = v_kite_body - v_wind_body
    
    airspeed = np.linalg.norm(v_apparent)
    alpha_rad = np.arctan2(v_apparent[2], v_apparent[0])
    alpha_deg = np.degrees(alpha_rad)
    return v_kite_body, airspeed, alpha_deg, alpha_rad

def get_rotation_matrix(gamma_rad, psi_rad):
    # Rotation matrix for -90 degrees around the y-axis
    R_y90 = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [-1, 0, 0]])
    return R_y90

   
    R_ygamma = np.array([[np.cos(-gamma_rad), 0, -np.sin(-gamma_rad)],
                         [0, 1, 0],
                         [np.sin(-gamma_rad), 0, np.cos(-gamma_rad)]])
    
    # Rotation matrix for -psi around the z''-axis
    R_zpsi = np.array([[np.cos(-psi_rad), np.sin(-psi_rad), 0],
                        [-np.sin(-psi_rad), np.cos(-psi_rad), 0],
                        [0, 0, 1]])

    # Total rotation matrix is the product of these three rotations
    R_total = R_zpsi @ R_ygamma @ R_y90
    


def skew_matrix(w):
    Define the skew-symmetric matrix for the rotation vector [0, 0, -w]
    return np.array([[0, 0, 0],
                     [0, 0, -w],
                     [0, w, 0]])


def update_rotation_matrix(R0, w):
    Updates the rotation matrix using the skew-symmetric matrix and the initial R0.
    skew_w = skew_matrix(w)
    # Cross product to update rotation matrix
    R_new = R0 @ skew_w
    return R_new
"""

# --- Sampling ---
samples, coords_grid, coords_phys = helical_sample_velocity_field_physical_moving_box(
    data_u, data_v, data_w, U_box = 8.5,
    radius_m=60,
    T_loop=10,
    total_time=330,
    DX=0.3632, DY=7.5, DZ=7.5
)

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


# Store the values for plotting
gamma_rad = np.arcsin(dz / v_mag)
gamma_deg = np.degrees(gamma_rad)
DY = 7.5
DZ = 7.5
y_offset = (Ny * DY) / 2
z_offset = (Nz * DZ) / 2
phi_rad = np.arctan2(z - z_offset, y - y_offset)
phi_deg = np.degrees(phi_rad)

# --- Initialise list to store components ---
airspeeds = []
apparent_wind_angle_deg = []  
v_body_x = []
v_body_y = []
v_body_z = []

# Initialize R0 with the first rotation matrix (for the first timestep)
#R0 = get_rotation_matrix(gamma_rad[0], phi_rad[0])

for i in range(len(t)):
    v_kite_global_i = v_global[i]
    v_windglobal_i = np.array([8.5, 0, 0]) + samples[i, :]

    # --- Step 1: Tangential direction of motion (X_body)
    X_body = v_kite_global_i / np.linalg.norm(v_kite_global_i)

    # --- Step 2: Radial vector from position to helix center in YZ
    # Since helix is in YZ plane and center is at y_offset, z_offset
    radial_vec = np.array([0, y_offset - y[i], z_offset - z[i]])
    radial_vec_outward = -radial_vec 
    Y_body = radial_vec_outward / np.linalg.norm(radial_vec_outward)

    # --- Step 3: Z_body = Cross product (right-hand rule)
    Z_body = np.cross(X_body, Y_body)
    Z_body /= np.linalg.norm(Z_body)

    # --- Re-orthonormalize Y_body = Z x X (more robust)  [Fixes edges]
    Y_body = np.cross(Z_body, X_body)
    Y_body /= np.linalg.norm(Y_body)

    # --- Step 4: Rotation matrix inertial -> body
    R_nb = np.column_stack((X_body, Y_body, Z_body))  # Columns: [X Y Z]
    # --- Step 5: Transform velocities into body frame
    v_kite_body = R_nb.T @ v_kite_global_i
    v_wind_body = R_nb.T @ v_windglobal_i
    v_apparent = v_kite_body - v_wind_body

    airspeed = np.linalg.norm(v_apparent)
    awa_rad = np.arctan2(v_apparent[2], v_apparent[0])  # AoA = Z over X
    awa_deg = np.degrees(awa_rad)

    # Store results
    airspeeds.append(airspeed)
    apparent_wind_angle_deg.append(awa_deg)
    v_body_x.append(v_kite_body[0])
    v_body_y.append(v_kite_body[1])
    v_body_z.append(v_kite_body[2])


# Convert lists to arrays for later use
airspeeds = np.array(airspeeds)
apparent_wind_angle_deg = np.array(apparent_wind_angle_deg)
v_body_x = np.array(v_body_x)
v_body_y = np.array(v_body_y)
v_body_z = np.array(v_body_z)


# --- Plotting ---

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

plt.figure(figsize=(10, 5))
plt.plot(t, gamma_deg, label='Tilt angle γ [deg]')
plt.plot(t, phi_deg, label='Circumferential angle φ [deg]')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Angle [deg]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('angles_gamma_phi.pdf')

plt.figure(figsize=(10, 5))
plt.plot(t[1:-1], airspeeds[1:-1], label='Airspeed [m/s]', color='blue')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Value', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(t[1:-1], apparent_wind_angle_deg[1:-1], label='Apparent Wind Angle [deg]', color='orange')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Value', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('aoa_crossproductmatrix.pdf')


# Plot Body Frame velocity components
plt.figure(figsize=(10, 6))
plt.plot(t, v_body_x, label='v_x (Body Frame)', color='blue')
plt.plot(t, v_body_y, label='v_y (Body Frame)', color='orange')
plt.plot(t, v_body_z, label='v_z (Body Frame)', color='green')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Velocity Components in the Body Frame')
plt.legend()
plt.grid(True)
plt.tight_layout()

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
plt.show()

