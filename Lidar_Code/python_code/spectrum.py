import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions_spectrum import *

def main():

    #Initial setup
    lidar_data = pd.read_csv('modified_lidar_data21_2025-02-13_113639.csv') #Import data files
    timestamp = lidar_data['Timestamp (s)']-808012801
    height = 100 #set desired height column
    fs = (1/11) #Sample frequency in Hz

      # Construct the column name based on i
    col_name = f'Horizontal Wind Speed (m/s) at {height}m'

    if col_name not in lidar_data.columns:
        raise ValueError(f"Column '{col_name}' not found in the dataset!")

    # Apply the interpolation function on the 'Horizontal Wind Speed (m/s) at 300m' column
    lidar_data['Horizontal Wind Speed (m/s) at 300m'] = interpolate_with_threshold(lidar_data['Horizontal Wind Speed (m/s) at 300m'])
    print(len(lidar_data['Horizontal Wind Speed (m/s) at 300m']))
    # Create a mask to identify non-NaN values
    mask = ~np.isnan(lidar_data[col_name])
    # Ensure the mask size is correct
    mask_size = np.sum(mask)  # Count the number of True values in the mask
    print(mask_size)
    # Get the valid data (excluding NaNs)
    valid_data = lidar_data[col_name][mask]

    # Get spectrum
    freqs_u, psd_u = compute_psd(valid_data, fs)

    # Smoothing
    n_per_decade = 20  # Set to a value between ~10 and 20
    smoothed_freqs, smoothed_spectrum = smooth_criminal(psd_u, freqs_u, n_per_decade)

    # Sanity check for spectrum
    speccheck_unsmooth(f"Lidar Data {height}m", lidar_data[col_name].dropna(), freqs_u, psd_u)
    speccheck_smooth(f"Lidar Data {height}m", lidar_data[col_name].dropna(), smoothed_freqs, smoothed_spectrum)

    
    # Compute acceleration in the time domain (I think this works)
    spectral_accel_time = compute_spectral_acceleration_time(valid_data, fs) 

    # Now, restore NaNs back to their original positions in the time-domain result
    # Pad the inverse FFT result to match the original size of lidar_data[col_name]
    spectral_accel_time_full = np.full_like(lidar_data[col_name], np.nan)  # Start with zeros (this will preserve the NaNs)
    
    # Insert the spectral acceleration values into the valid positions (where mask is True)
    spectral_accel_time_full[mask] = spectral_accel_time[:len(valid_data)]  # Match size by taking the first N elements

    #Finite difference
    finite_accel = central_diff_acceleration(lidar_data[col_name], 11)
    # Create 3-minute windows for the acceleration time series
    window_size = int(180 / 11)  # Number of samples per 3-minute window
    #window_size1 = int(600/11)
    # Compute the percentiles (P90) for each 3-minute window
    percentiles_values = compute_percentiles(spectral_accel_time_full, window_size, percentiles=[90])
    percentiles_finite = compute_percentiles(finite_accel, window_size, percentiles=[90] )
    #percentiles_values1 = compute_percentiles(spectral_accel_time_full, window_size1, percentiles=[90])
    # P90 accelerations
    p90_values = percentiles_values[90]
    p90_values1 = percentiles_finite[90]

    #Stats on P90 accel
    mean_p90 = np.mean(p90_values)
    std_p90 = np.std(p90_values)
  
    #PLOTTING
    #Spectrum
    plt.figure(figsize=(10, 6))
    plt.loglog(freqs_u, psd_u, color = 'blue', label = 'u')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density')
    plt.legend()
    plt.grid(True)
    
    #Smoothed spectrum
    plt.figure(figsize=(10, 6))
    plt.loglog(freqs_u, psd_u, label='Raw Spectrum', alpha=0.5)
    plt.loglog(smoothed_freqs, smoothed_spectrum, label='Smoothed Spectrum', linewidth=2)
    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density')
    
    
    #Acceleration time domain
    plt.figure(figsize=(10, 6))
    plt.plot(timestamp/3600, spectral_accel_time_full, label='Spectral Acceleration (Time Domain)', color='red', alpha=0.7)
    plt.plot(timestamp/3600, finite_accel, label='Finite Acceleration (Time Domain)', color='green', alpha=0.7)
    plt.xlabel('Time [hr]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.legend()
    plt.grid(True)
    plt.title('Acceleration in Time Domain')



    # 90th Percentile Histogram (Acceleration on x-axis)
    plt.figure(figsize=(10, 6))
    plt.hist(p90_values, bins=40, color='green', alpha=0.7, edgecolor='black', label ='spectralaccel')  # Customize the number of bins
    plt.hist(p90_values1, bins=40, color='blue', alpha=0.7, edgecolor='black', label = 'finiteaccel' )
    plt.xlabel('Acceleration (m/s²)')  # X-axis label (acceleration)
    plt.ylabel('Frequency')  # Y-axis label (how often each acceleration value occurs)
    plt.title('Histogram of 90th Percentile Acceleration')
    plt.legend()
    plt.grid(True)

    plt.show()

    
    
    
main()