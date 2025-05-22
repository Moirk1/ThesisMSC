import numpy as np
from scipy.ndimage import map_coordinates


def linear_sample_velocity_field(data_u, data_v, data_w,
                                  U_box, total_time=300, DX=0.3632, DY=7.5, DZ=7.5,
                                  y_fixed=None, z_fixed=None):

    Nx, Ny, Nz = data_u.shape
    dt = 0.1
    n_samples = int(total_time / dt)
    t = np.linspace(0, total_time, n_samples)
    
    U_kite_box = (1/3 * U_box)
    x_phys = (U_box - U_kite_box) * t  

    # If not given, sample through the center of the box
    if y_fixed is None:
        y_fixed = (Ny * DY) / 2
    if z_fixed is None:
        z_fixed = (Nz * DZ) / 2

    x_grid = x_phys / DX
    y_grid = np.full_like(x_grid, y_fixed / DY)
    z_grid = np.full_like(x_grid, z_fixed / DZ)

    sample_coords_grid_linear = np.vstack((x_grid, y_grid, z_grid)).T
    sample_coords_phys_linear = np.vstack((x_phys, y_fixed * np.ones_like(t), z_fixed * np.ones_like(t))).T

    u_vals = map_coordinates(data_u, sample_coords_grid_linear.T, order=1, mode='nearest')
    v_vals = map_coordinates(data_v, sample_coords_grid_linear.T, order=1, mode='nearest')
    w_vals = map_coordinates(data_w, sample_coords_grid_linear.T, order=1, mode='nearest')

    linear_samples = np.vstack((u_vals, v_vals, w_vals)).T
    print("Linear samples:", linear_samples.shape)
    return linear_samples, sample_coords_grid_linear, sample_coords_phys_linear

# Samples a turbulent velocity field along a helical trajectory through a moving wind box
def helical_sample_velocity_field_physical_moving_box(data_u, data_v, data_w,
                                                      U_box, radius_m=60, T_loop=10, 
                                                      total_time=300, DX=0.3632, DY=7.5, DZ=7.5):

    Nx, Ny, Nz = data_u.shape
    U_kite_box = (1/3 * U_box)
    dt = 0.1  # seconds per sample
    n_samples = int(total_time / dt)
    t = np.linspace(0, total_time, n_samples)
   
    x_phys = (U_box - U_kite_box) * t
    
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