import numpy as np

#Computes aerodynamic forces and body-frame velocities of a kite moving through a turbulent wind field.
def compute_aerodynamics(t, x, y, z, v_global, samples,
                         U_mean, rho, A, CD0, a, DY, DZ, Ny, Nz,
                         fix_aoa_deg=None, pitch_reference=None):
    airspeeds = []
    apparent_wind_angle_deg = []
    lift = []
    drag = []
    F_aero = []
    F_aero_vec_list = []
    aoa_deg_corrected = []
    lift_coeff = []
    drag_coeff = []

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
        #C_D_tether = C_perp * (d_tether * L_tether) / (4 * A_tether)
        C_D_tether = 0
        C_D_total = C_D_airfoil + C_D_tether
       
        #Magnitude of Lift & Drag
        q = 0.5 * rho * airspeed**2
        L = q * A * C_L
        D = q * A * C_D_total

        #Lift and Drag Vector 
        v_apparent_unit = v_apparent / airspeed
        drag_vec = -v_apparent_unit
        lift_dir = np.cross(drag_vec, Y_body)
        lift_dir /= np.linalg.norm(lift_dir)
        lift_vec = L * lift_dir
        drag_vec = D * drag_vec

        #Aerodynamic force vector and mag
        F_aero_vec = lift_vec + drag_vec
        F_a = np.sqrt(L**2 + D**2)

        # Store
        airspeeds.append(airspeed)
        apparent_wind_angle_deg.append(awa_deg)
        aoa_deg_corrected.append(aoa_deg)
        lift.append(L)
        drag.append(D)
        F_aero.append(F_a)
        F_aero_vec_list.append(F_aero_vec)
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
        'F_aero_vec': np.array(F_aero_vec_list),
        'v_body': np.vstack((v_body_x, v_body_y, v_body_z)).T
    }