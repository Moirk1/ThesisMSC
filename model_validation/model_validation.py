import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---CONSTANT DEFINITIONS---
rho = 1.225  # Air density (kg/m³)
A =  2.982      # Kite area (m²)
CD0 = 0.004  # Minimum drag coefficient
a = 0.008    # Drag coefficient scaling factor

#----FUNCTION DEFINITIONS ----------
def quaternion_to_euler(q_nb):
    """
    Convert quaternion to Euler angles (degrees)
    Assumes q_nb = [q4, q1, q2, q3] where q4 is scalar
    """
    q1 = q_nb[1]
    q2 = q_nb[2]
    q3 = q_nb[3]
    q4 = q_nb[0]

    # Euler angles in radians
    phi = np.arctan2(2*(q1*q4 + q2*q3), 1 - 2*(q1**2 + q2**2)) #roll
    theta = np.arcsin(np.clip(2*(q2*q4 - q1*q3), -1.0, 1.0)) #pitch
    psi = np.arctan2(2*(q3*q4 + q1*q2), 1 - 2*(q2**2 + q3**2)) #yaw

    return np.array([phi, theta, psi])  # radians

def rotation_matrix_from_euler(phi, theta, psi):
    """
    Create rotation matrix from Euler angles (radians)
    Converts vector from NED to body frame
    """
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    R = np.array([
        [c_theta * c_psi, c_theta * s_psi, -s_theta],
        [-c_phi * s_psi + s_phi * s_theta * c_psi,  c_phi * c_psi + s_phi * s_theta * s_psi, s_phi * c_theta],
        [s_phi * s_psi + c_phi * s_theta * c_psi, -s_phi * c_psi + c_phi * s_theta * s_psi, c_phi * c_theta]
    ])
    return R


def calculate_airspeed_and_aoa(v_body, wind_ned, q_nb):
    """
    Calculate airspeed (magnitude) and angle of attack.
    v_body: velocity of the aircraft in the body frame
    wind_ned: wind velocity in the NED frame
    q_nb: quaternion [eta_nb, epsilon_1_nb, epsilon_2_nb, epsilon_3_nb]
    """
    # Step 1: Convert wind velocity from NED to body frame using the rotation matrix
    euler_angles = quaternion_to_euler(q_nb)  # Convert quaternion to Euler angles
    R = rotation_matrix_from_euler(*euler_angles)  # Get the rotation matrix

    # Wind components in the NED frame
    wind_north = wind_ned[0]
    wind_east = wind_ned[1]
    wind_down = wind_ned[2]

    # Wind velocity vector in NED frame
    wind_ned_vector = np.array([wind_north, wind_east, wind_down])

    # Convert wind velocity from NED to body frame
    wind_body = R @ wind_ned_vector  # Matrix multiplication to rotate wind vector

    # Step 2: Calculate apparent wind velocity in body frame
    apparent_wind_body = v_body - wind_body

    # Step 3: Calculate airspeed (magnitude of apparent wind velocity)
    airspeed = np.linalg.norm(apparent_wind_body)

    #Step 4: Calculate angle of attack (α) [Assuming no pitch]
    v_a_zb = apparent_wind_body[2]  # Apparent wind component along the z-axis (down)
    v_a_xb = apparent_wind_body[0]  # Apparent wind component along the x-axis (forward)

    alpha = np.arctan2(v_a_zb, v_a_xb)
    alpha_deg = np.rad2deg(alpha)   # Convert angle of attack from radians to degrees

    return airspeed, alpha_deg, alpha

def compute_aero_forces(V_rel, alpha, A, rho, C_D0, a):
    """
    Computes lift, drag and aerodynamic forces on the aircraft (no tether drag is included)
    
    """
    # Lift coefficient from angle of attack
    C_L = 2 * np.pi * alpha
    
    # Airfoil drag coefficient
    C_D_airfoil = C_D0 + a * C_L**2

    # Dynamic pressure
    q = 0.5 * rho * V_rel**2

    # Lift and Drag forces
    L = q * A * C_L
    D = q * A * C_D_airfoil

    #Aerodynamic Force
    F_a = np.sqrt(L**2 + D**2)

    return C_L, C_D_airfoil, L, D, F_a


#----MAIN SECTION -----    
# Load your CSV data
simulation_data = pd.read_csv("simulation_results.csv")
timestamp = simulation_data['time']

# Initialize lists to store calculated and real values
calculated_airspeed = []
calculated_aoa_deg = [] 
calculated_Cl = []
calculated_CD = []
calculated_lift = []
calculated_drag = []
calculated_Fa = []
real_airspeed = simulation_data['V_a'].values  # Real airspeed from the dataset
real_aoa = simulation_data['alpha'].values  # Real angle of attack from the dataset


# Loop through each row in the DataFrame and extract the relevant values
for index, row in simulation_data.iterrows():
    # Extract the necessary data for each row
    v_body = np.array([row['u'], row['v'], row['w']])  # Aircraft velocity in body frame
    wind_ned = np.array([row['wind_north'], row['wind_east'], row['wind_down']])  # Wind velocity in NED frame
    q_nb = np.array([row['eta_nb'], row['epsilon1_nb'], row['epsilon2_nb'], row['epsilon3_nb']])  # Quaternion

    # Calculate airspeed for this row
    V_a, alpha_deg, alpha = calculate_airspeed_and_aoa(v_body, wind_ned, q_nb)
    C_L, C_D_airfoil, L, D, F_a = compute_aero_forces(V_a, alpha, A, rho, CD0, a)

    #Append values to list
    calculated_airspeed.append(V_a)
    calculated_aoa_deg.append(alpha_deg)
    calculated_Cl.append(C_L)
    calculated_CD.append(C_D_airfoil)
    calculated_lift.append(L)
    calculated_drag.append(D)
    calculated_Fa.append(F_a)
   

# Convert lists to numpy arrays for easier plotting
calculated_airspeed = np.array(calculated_airspeed)
calculated_aoa_deg = np.array(calculated_aoa_deg)
calculated_Cl= np.array(calculated_Cl)
calculated_Cd = np.array(calculated_CD)
calculated_lift = np.array(calculated_lift)
calculated_drag = np.array(calculated_drag)
calculated_Fa = np.array(calculated_Fa)


#-------------- PLOTTING -------------------
#Airspeed plot
plt.figure(figsize=(8, 6))
plt.plot(timestamp, real_airspeed, label='Real Airspeed', color='blue', linestyle='-')
plt.plot(timestamp, calculated_airspeed, label='Calculated Airspeed', color='red', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Airspeed [m/s]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('airspeed_real_vs_calculated_sim.pdf')

# Angle of attack plot
plt.figure(figsize=(8, 6))
plt.plot(timestamp, real_aoa, label='Real AoA', color='blue', linestyle='-')
plt.plot(timestamp, calculated_aoa_deg, label='Calculated AoA', color='red', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Angle of Attack [degrees]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('aoa_real_vs_calculated_sim.pdf')

# Plot C_L and C_D vs AOA
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axs[0].plot(calculated_aoa_deg, calculated_Cl, label='Calculated C_L', color='blue', linestyle='-')
axs[0].set_ylabel('C_L [-]')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(calculated_aoa_deg, calculated_CD, label='Calculated C_D', color='red', linestyle='-')
axs[1].set_xlabel('AOA [deg]')
axs[1].set_ylabel('C_D [-]')
axs[1].legend()
axs[1].grid(True)
plt.tight_layout()

# Plot Lift and Drag vs time
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axs[0].plot(timestamp, calculated_lift, label='Calculated Lift', color='blue', linestyle='-')
axs[0].set_ylabel('Lift [N]')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(timestamp, calculated_drag, label='Calculated Drag', color='red', linestyle='-')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Drag [N]')
axs[1].legend()
axs[1].grid(True)
plt.tight_layout()

# Plot Aerodynamic Force vs time
plt.figure(figsize=(8, 6))
plt.plot(timestamp, calculated_Fa, label='Aerodynamic Force', color='blue', linestyle='-')
plt.xlabel('Time [s]')
plt.ylabel('Aerodynamic Force [N]')
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()