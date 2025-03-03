import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions_spectrum import *

def main():

    #Initial setup
    lidar_data = pd.read_csv('modified_lidar_data21_2025-02-13_113639.csv') #Import data files
    fs = (1/11) #Sample frequency in Hz

    #Get spectrum
    freqs_u, psd_u = compute_psd(lidar_data['Horizontal Wind Speed (m/s) at 250m'].dropna(), fs) 
    
    #Smoothing
    n_per_decade = 20  # Set to a value between ~10 and 20
    smoothed_freqs, smoothed_spectrum = smooth_criminal(psd_u, freqs_u, n_per_decade) #Smoothed spectrum

    #Sanity check for spectrum
    speccheck_unsmooth("Lidar Data 250m", lidar_data['Horizontal Wind Speed (m/s) at 250m'].dropna(), freqs_u, psd_u)
    speccheck_smooth("Lidar Data 250m", lidar_data['Horizontal Wind Speed (m/s) at 250m'].dropna(), smoothed_freqs, smoothed_spectrum)
    
    #Compute spectral acceleration from the wind speed power spectrum
    spectral_accel_freq = compute_spectral_acceleration(freqs_u, psd_u)
    #Apply smoothing to spectral acceleration
    smoothed_accel_freqs, smoothed_accel_spectrum = smooth_criminal(spectral_accel_freq, freqs_u, n_per_decade)
    
    # Compute acceleration in the time domain (NEEDS WORK)
    spectral_accel_time = compute_spectral_acceleration_time(psd_u, freqs_u) 
   

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
    plt.plot(spectral_accel_time[1:], label='Spectral Acceleration (m/s²)', color='red', alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.title('Wind Speed and Spectral Acceleration in Time Domain')
    plt.show()
    
    
    
    
main()