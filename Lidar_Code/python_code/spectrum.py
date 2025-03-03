import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions_spectrum import *

def main():

#Implement check that np.trapz(Su(f)) = 0.5 * sigma_u**2, area under spectrum == 0.5 variance

    lidar_data = pd.read_csv('modified_lidar_data21_2025-02-13_113639.csv') #import hovsore data file
    ##1.3 pg 72 in book describes the process for ensemble average
    fs = (1/11) #sample frequency in Hz

#Unsmoothed Calculation (using function defined in functions)
    freqs_u, psd_u = compute_psd(lidar_data['Horizontal Wind Speed (m/s) at 250m'].dropna(), fs) #So there must be nans somewhere in the u column cause if I don't put it in the graph is empty maybe could pre process
    
    n_per_decade = 20  # Set to a value between ~10 and 20
    smoothed_freqs, smoothed_spectrum = smooth_criminal(psd_u, freqs_u, n_per_decade)

    speccheck_unsmooth("Lidar Data 250m", lidar_data['Horizontal Wind Speed (m/s) at 250m'].dropna(), freqs_u, psd_u)
    speccheck_smooth("Lidar Data 250m", lidar_data['Horizontal Wind Speed (m/s) at 250m'].dropna(), smoothed_freqs, smoothed_spectrum)
     # Compute spectral acceleration from the wind speed power spectrum
    spectral_accel_freq = compute_spectral_acceleration(freqs_u, psd_u)
     # Apply smoothing to spectral acceleration
    smoothed_accel_freqs, smoothed_accel_spectrum = smooth_criminal(spectral_accel_freq, freqs_u, n_per_decade)
    
    # Compute acceleration in the time domain
    spectral_accel_time = compute_spectral_acceleration_time(psd_u, freqs_u) #this works but I need to sort time
   
    plt.figure(figsize=(10, 6))
    plt.loglog(freqs_u, psd_u, color = 'blue', label = 'u')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density')
    plt.legend()
    plt.grid(True)
    

    plt.figure(figsize=(10, 6))
    # Plot theoretical curve
    plt.loglog(freqs_u, psd_u, label='Raw Spectrum', alpha=0.5)
    plt.loglog(smoothed_freqs, smoothed_spectrum, label='Smoothed Spectrum', linewidth=2)
    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density')
    

    # Plot the spectral acceleration in the frequency domain
    plt.figure(figsize=(10, 6))
    plt.loglog(freqs_u, spectral_accel_freq, color='red', label='Spectral Acceleration (Freq Domain)')
    plt.loglog(smoothed_accel_freqs, smoothed_accel_spectrum, label='Smoothed Spectral Acceleration', linewidth=2, color='darkred')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Spectral Acceleration [m²/s²/Hz]')
    plt.legend()
    plt.grid(True)
    plt.title('Spectral Acceleration in Frequency Domain')
    plt.show()
    
    """
    plt.figure(figsize=(10, 6))
    plt.plot(spectral_accel_time, label='Spectral Acceleration (m/s²)', color='red', alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.title('Wind Speed and Spectral Acceleration in Time Domain')
    plt.show()
    """
    
    
    
main()