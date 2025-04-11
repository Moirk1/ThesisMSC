import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
rho = 1.225  # Air density (kg/m³)
A = 3       # Kite area (m²)
CD0 = 0.004  # Minimum drag coefficient
a = 0.008    # Drag coefficient scaling factor

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

    # Convert angle of attack from radians to degrees
    alpha_deg = np.rad2deg(alpha)

    return airspeed, alpha_deg

    
# Load your CSV data
simulation_data = pd.read_csv("simulation_results.csv")
timestamp = simulation_data['time']

# Initialize lists to store calculated and real values
calculated_airspeed = []
calculated_aoa = []  # Angle of attack
real_airspeed = simulation_data['V_a'].values  # Real airspeed from the dataset
real_aoa = simulation_data['alpha'].values  # Real angle of attack from the dataset


# Loop through each row in the DataFrame and extract the relevant values
for index, row in simulation_data.iterrows():
    # Extract the necessary data for each row
    v_body = np.array([row['u'], row['v'], row['w']])  # Aircraft velocity in body frame
    wind_ned = np.array([row['wind_north'], row['wind_east'], row['wind_down']])  # Wind velocity in NED frame
    q_nb = np.array([row['eta_nb'], row['epsilon1_nb'], row['epsilon2_nb'], row['epsilon3_nb']])  # Quaternion

    # Calculate airspeed for this row
    V_a, alpha_deg = calculate_airspeed_and_aoa(v_body, wind_ned, q_nb)
    calculated_airspeed.append(V_a)
    calculated_aoa.append(alpha_deg)
   
# Convert lists to numpy arrays for easier plotting
calculated_airspeed = np.array(calculated_airspeed)
calculated_aoa = np.array(calculated_aoa)

#Airspeed plot
plt.figure(figsize=(8, 6))
plt.plot(timestamp, real_airspeed, label='Real Airspeed', color='blue', linestyle='-')
plt.plot(timestamp, calculated_airspeed, label='Calculated Airspeed', color='red', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Airspeed (m/s)')
plt.title('Real vs Calculated Airspeed')
plt.legend()
plt.tight_layout()

# Angle of attack plot
plt.figure(figsize=(8, 6))
plt.plot(timestamp, real_aoa, label='Real AoA', color='green', linestyle='-')
plt.plot(timestamp, calculated_aoa, label='Calculated AoA', color='orange', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Angle of Attack (degrees)')
plt.title('Real vs Calculated Angle of Attack')
plt.legend()
plt.tight_layout()

plt.show()