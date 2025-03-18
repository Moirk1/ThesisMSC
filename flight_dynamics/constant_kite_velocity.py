import numpy as np
import matplotlib.pyplot as plt

# Constants
rho = 1.225  # Air density (kg/m^3)
A = 32.9      # Kite area (m^2)
CD0 = 0.02   # Minimum drag coefficient
a = 0.1      # Drag coefficient scaling constant

# Kite velocity (assumed constant)
v_kx = 10  # Horizontal kite speed (m/s)
v_ky = 5   # Vertical kite speed (m/s)

# Wind speed range (simulating variations)
wind_speeds = np.linspace(8, 20, 100)  # Wind speeds from 8 to 20 m/s

# Storage for results
angles_of_attack = []
lift_forces = []
drag_forces = []
tension_forces = []
power_harvested = []

# Loop over different wind speeds
for v_w in wind_speeds:
    # Compute apparent wind velocity components
    v_ax = v_w - v_kx
    v_ay = -v_ky
    
    # Compute apparent wind speed
    v_a = np.sqrt(v_ax**2 + v_ay**2)
    
    # Compute angle of attack
    alpha = np.arctan2(v_ax, -v_ay)  # In radians
    angles_of_attack.append(np.degrees(alpha))  # Store in degrees
    
    # Compute aerodynamic coefficients
    CL = 2 * np.pi * alpha  # Lift coefficient
    CD = CD0 + a * CL**2    # Drag coefficient
    CR = np.sqrt(CL**2+CD**2)
    # Compute forces
    L = 0.5 * CL * rho * A * v_a**2
    D = 0.5 * CD * rho * A * v_a**2
    
    # Compute tension force
    Ft = np.sqrt(L**2 + D**2)

    # Compute the wing speed ratio λ
    lambda_w = v_a / v_w
    
    # Compute power harvesting factor
    zeta = (lambda_w**2 * CR - lambda_w**3 * CD)
    
    
    # Store forces
    lift_forces.append(L)
    drag_forces.append(D)
    tension_forces.append(Ft)
    power_harvested.append(zeta)


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
