import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions_spectrum import *

def main():

    #Initial setup
    lidar_data = pd.read_csv('modified_lidar_data21_2025-02-13_113639.csv') #Import data files
    timestamp = lidar_data['Timestamp (s)']-808012801
    height = 300 #set desired height column
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

    #Compute spectral acceleration from the wind speed power spectrum
    spectral_accel_freq = compute_spectral_acceleration(freqs_u, psd_u)
    #Apply smoothing to spectral acceleration
    smoothed_accel_freqs, smoothed_accel_spectrum = smooth_criminal(spectral_accel_freq, freqs_u, n_per_decade)
    
    # Compute acceleration in the time domain (I think this works)
    spectral_accel_time = compute_spectral_acceleration_time(valid_data, fs) 

    # Now, restore NaNs back to their original positions in the time-domain result
    # Pad the inverse FFT result to match the original size of lidar_data[col_name]
    spectral_accel_time_full = np.full_like(lidar_data[col_name], np.nan)  # Start with zeros (this will preserve the NaNs)
    
    # Insert the spectral acceleration values into the valid positions (where mask is True)
    spectral_accel_time_full[mask] = spectral_accel_time[:len(valid_data)]  # Match size by taking the first N elements
   
    # Create 3-minute windows for the acceleration time series
    window_size = int(180 / 11)  # Number of samples per 3-minute window
    
    # Compute the percentiles (P75 and P90) for each 3-minute window
    percentiles_values = compute_percentiles(spectral_accel_time_full, window_size, percentiles=[75, 90])

    # P75 and P90 percentiles
    p75_values = percentiles_values[75]
    p90_values = percentiles_values[90]
    
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
    
    #Spectral Acceleration (frequency domain)
    plt.figure(figsize=(10, 6))
    plt.loglog(freqs_u, spectral_accel_freq, color='red', label='Spectral Acceleration (Freq Domain)')
    plt.loglog(smoothed_accel_freqs, smoothed_accel_spectrum, label='Smoothed Spectral Acceleration', linewidth=2, color='darkred')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Spectral Acceleration [m²/s²/Hz]')
    plt.legend()
    plt.grid(True)
    plt.title('Spectral Acceleration in Frequency Domain')
    
    #Spectral acceleration time domain
    plt.figure(figsize=(10, 6))
    plt.plot(timestamp[2:], spectral_accel_time_full[2:], label='Spectral Acceleration (Time Domain)', color='red', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.title('Wind Speed and Spectral Acceleration in Time Domain')


    plt.show()

    
    
    
main()