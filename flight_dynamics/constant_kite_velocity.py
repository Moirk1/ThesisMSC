import numpy as np
import matplotlib.pyplot as plt

# Constants
rho = 1.225  # Air density (kg/m^3)
A = 3      # Kite area (m^2) Taken from Roland Schmehl lectures
CD0 = 0.004   # Minimum drag coefficient (Both these values are NACA 2412)
a = 0.008    # Drag coefficient scaling constant

#HERE WILL SETUP SOMETHING TO TAKE IN DATA 
#Airspeed (Placeholders)
v_airspeed = 14
#VNED frame (Placeholders)
v_n = 12
v_e = 4
#Yaw angle 
psi = np.radians(10)

#Ground reference frame
v_ground = np.sqrt(v_n^2+v_e^2)
v_w = v_ground-v_airspeed

#Find velocity of kite 
v_kx = v_airspeed*np.sin(psi)
v_ky = v_airspeed*np.cos(psi)

# Compute apparent wind velocity components
v_ax = v_w - v_kx
v_ay = -v_ky
    
# Compute apparent wind speed
v_a = np.sqrt(v_ax**2 + v_ay**2)
    
# Compute angle of attack
alpha = np.arctan2(v_ax, -v_ay)  # In radians
#angles_of_attack.append(np.degrees(alpha))  # Store in degrees
    
# Compute aerodynamic coefficients
CL = 2 * np.pi * alpha  # Lift coefficient
CD = CD0 + a * CL**2    # Drag coefficient
CR = np.sqrt(CL**2+CD**2)
# Compute forces
L = 0.5 * CL * rho * A * v_a**2
D = 0.5 * CD * rho * A * v_a**2
    
# Compute tension force
Ft = np.sqrt(L**2 + D**2)

#Compute the wing speed ratio λ
lambda_w = v_a / v_w
    
# Compute power harvesting factor
zeta = (lambda_w**2 * CR - lambda_w**3 * CD)


    
"""
# Plot results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(wind_speeds, angles_of_attack, label="Angle of Attack (deg)")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Angle of Attack (degrees)")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(wind_speeds, lift_forces, label="Lift (N)")
plt.plot(wind_speeds, drag_forces, label="Drag (N)")
plt.plot(wind_speeds, tension_forces, label="Tension Force (N)")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Force (N)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
# Forces and power plot
plt.plot()
plt.plot(wind_speeds, power_harvested, label="Power Harvested (W)", linestyle='--')
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Power Harvesting Factor [-]")
plt.legend()
plt.grid()


plt.show()
"""