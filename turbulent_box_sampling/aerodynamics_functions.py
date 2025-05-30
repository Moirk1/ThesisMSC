import numpy as np
from scipy.ndimage import label

#Computes aerodynamic forces and body-frame velocities of a kite moving through a turbulent wind field.
def compute_aerodynamics(t, x, y, z, v_global, samples,
                         U_mean, rho, A, CD0, a, DY, DZ, Ny, Nz,
                         fix_aoa_deg=None, pitch_reference=None):
    airspeeds = []
    apparent_wind_angle_deg = []
    lift = []
    lift_vec_list =[]
    drag = []
    drag_vec_list = []
    F_aero = []
    F_aero_vec_list = []
    aoa_deg_corrected = []
    lift_coeff = []
    drag_coeff = []
    v_apparent_vec_list = []

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
        v_apparent =   v_wind_body - v_kite_body
        
        airspeed = np.linalg.norm(v_apparent)
        awa_rad = np.arctan2(-v_apparent[2], -v_apparent[0])  # AoA
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
        #C_D_tether = C_perp * (d_tether * L_tether) / (4 * A_tether)
        C_D_tether = 0
        C_D_total = C_D_airfoil + C_D_tether
       
        #Magnitude of Lift & Drag
        q = 0.5 * rho * airspeed**2
        L = q * A * C_L
        D = q * A * C_D_total



        # Normalize drag direction (opposite of apparent wind)
        e_D = (v_apparent / np.linalg.norm(v_apparent))

        #Defining z direction
        z_ref = np.array([0, 0, -1])

        # Compute side force direction (perpendicular to drag and up)
        cross = np.cross(e_D, z_ref)
        norm = np.linalg.norm(cross)
       
        e_Y = cross/norm
        e_L = np.cross(e_Y, e_D)
        L_vec = L * e_L
        D_vec = D * e_D

        F_aero_vec = L_vec + D_vec
        F_a = np.sqrt(L**2 + D**2)

        # Store
        airspeeds.append(airspeed)
        apparent_wind_angle_deg.append(awa_deg)
        aoa_deg_corrected.append(aoa_deg)
        lift.append(L)
        drag.append(D)
        lift_vec_list.append(L_vec)
        drag_vec_list.append(D_vec)
        F_aero.append(F_a)
        F_aero_vec_list.append(F_aero_vec)
        v_apparent_vec_list.append(v_apparent)
        v_body_x.append(v_kite_body[0])
        v_body_y.append(v_kite_body[1])
        v_body_z.append(v_kite_body[2])

    return {
        'airspeed': np.array(airspeeds),
        'awa_deg': np.array(apparent_wind_angle_deg),
        'aoa_deg': np.array(aoa_deg_corrected),
        'lift': np.array(lift),
        'drag': np.array(drag),
        'drag_vec':np.array(drag_vec_list),
        'lift_vec':np.array(lift_vec_list),
        'F_aero': np.array(F_aero),
        'F_aero_vec': np.array(F_aero_vec_list),
        'v_apparent_vec': np.array(v_apparent_vec_list),
        'v_body': np.vstack((v_body_x, v_body_y, v_body_z)).T
    }




def find_aoa_exceedances(aoa_deg, dt, threshold=10, min_duration=5):
    """
    Find time periods where AoA exceeds the threshold and lasts longer than min_duration.

    Parameters:
    - aoa_deg: numpy array of AoA in degrees
    - dt: time step in seconds
    - threshold: AoA threshold (default 10 degrees)
    - min_duration: minimum duration in seconds (default 5 sec)

    Returns:
    - List of (start_time, end_time) tuples for exceedance events
    """
    condition = aoa_deg > threshold
    labeled_array, num_features = label(condition)
    
    events = []
    for i in range(1, num_features+1):
        indices = np.where(labeled_array == i)[0]
        duration = len(indices) * dt
        if duration >= min_duration:
            t_start = indices[0] * dt
            t_end = indices[-1] * dt
            events.append((t_start, t_end))
    
    return events
