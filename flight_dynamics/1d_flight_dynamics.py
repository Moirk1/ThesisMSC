import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Constants and Parameters (I'm using Makani Properties for aircraft mass and wing area)
rho = 1.225  # Air density (kg/m^3)
A = 32.9  # Wing area (m^2)
C_D0 = 0.05  # Minimum drag coefficient
a = 0.1  # Drag coefficient scaling factor
m = 1730.8  # Aircraft mass (kg)

# Initial conditions (Assumed constant to start with)
V_r = 6  # m/s (reel out speed)
V_w = 12 # m/s (wind speed)
beta = np.radians(10)  # Chord angle in radians (10 degrees)

# Function to calculate lift coefficient (C_L) based on angle of attack
def lift_coefficient(alpha):
    return 2 * np.pi * alpha

# Function to calculate drag coefficient (C_D) based on C_L
def drag_coefficient(C_L):
    return C_D0 + a * C_L**2

# Function to calculate forces (Lift and Drag)
def calculate_forces(V_rel, alpha):
    C_L = lift_coefficient(alpha)
    C_D = drag_coefficient(C_L)
    L = 0.5 * C_L * rho * A * V_rel**2  # Lift force
    D = 0.5 * C_D * rho * A * V_rel**2  # Drag force
    return L, D

# Function to calculate relative wind speed and angle of attack
def relative_wind_speed(V_w, V_r, V_k):
    V_rel_x = V_w - V_r
    V_rel_y = V_k  #IN THE DOCUMENT NEGATIVE BUT I THINK MAY BE POSITIVE
    V_rel = np.sqrt(V_rel_x**2 + V_rel_y**2)
    alpha = np.arctan2(V_rel_x, V_rel_y)  # Angle of attack
    return V_rel, alpha

# Steady-state model function 
def steady_state(V_k, V_w, V_r, beta):
    V_k = V_k[0]  # Extract scalar value from array if needed
    
    # Relative wind speed and angle of attack
    V_rel, alpha = relative_wind_speed(V_w, V_r, V_k)
    
    # Calculate lift and drag forces
    L, D = calculate_forces(V_rel, alpha)
    
    # Normal and tangential forces
    F_N = L * np.cos(alpha) + D * np.sin(alpha)
    F_T = -L * np.sin(alpha) + D * np.cos(alpha)
    
    # Force balance: Assume lateral force F_y = 0
    F_x = F_N * np.cos(beta) - F_T * np.sin(beta)
    F_y = F_N * np.sin(beta) + F_T * np.cos(beta)
    
    # Print the current F_y to track it
    #print(f"V_k = {V_k:.2f} m/s, F_y = {F_y:.2f} N")
    
    return F_y**2  # minimise squared lateral force

# Function to solve the steady-state problem and find the V_k that minimises Fy
#I assume I know Vw, Vr and beta
def steady_state_solver(V_w, V_r, beta):
    # Initial guess for the kite's crosswind velocity (V_k)
    initial_guess = 10  # m/s (initial guess for lateral velocity)
    bounds = [(0, 20)]
    options = {'maxiter': 1000, 'gtol': 1e-6,}
    # Minimize F_y (lateral force) to bring it as close to zero as possible
    result = opt.minimize(steady_state, initial_guess, args=(V_w, V_r, beta), method='L-BFGS-B', bounds=bounds, options = options)
    
    # Check if the optimization was successful
    if result.success:
        V_k_optimized = result.x[0]
        print(f"\nOptimization Successful: V_k = {V_k_optimized:.2f} m/s")
        
        # Calculate the forces (lift, drag, and F_x) at the optimized V_k
        V_rel, alpha = relative_wind_speed(V_w, V_r, V_k_optimized)  # Calculate relative wind and angle of attack
        L, D = calculate_forces(V_rel, alpha)  # Calculate lift and drag
        F_N = L * np.cos(alpha) + D * np.sin(alpha)  # Normal force
        F_T = -L * np.sin(alpha) + D * np.cos(alpha)  # Tangential force
        F_x = F_N * np.cos(beta) - F_T * np.sin(beta)  # Normal force in the x-direction
        F_y = F_N * np.sin(beta) + F_T * np.cos(beta)
        # Print the forces
        print(f"At V_k = {V_k_optimized:.2f} m/s:")
        print(f"Lift Force (L): {L:.2f} N")
        print(f"Drag Force (D): {D:.2f} N")
        print(f"Normal Force in X (F_x): {F_x:.2f} N")
        print(f'Fy is {F_y:.2f} N (This should be basically zero)')  #I'm having some problems getting back -0 but think it's fine
    else:
        print("\nOptimization failed.")
        V_k_optimized = initial_guess

    # Return the optimized velocity
    return V_k_optimized

# Call the steady-state solver
V_k_optimized = steady_state_solver(V_w, V_r, beta)
print(f"Optimized lateral velocity (V_k): {V_k_optimized:.2f} m/s")