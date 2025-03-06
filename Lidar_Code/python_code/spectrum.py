import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions_spectrum import *  # Assuming the functions are imported from this file

def main():
    # Initial setup
    lidar_data = pd.read_csv('modified_lidar_data21_2025-02-13_113639.csv')  # Import data files
    timestamp = lidar_data['Timestamp (s)'] - 808012801  # Adjust timestamp as required
    fs = 1 / 11  # Sample frequency in Hz
    heights = [300, 250, 200, 150, 100]  # List of heights to process

    for height in heights:
        # Construct the column name based on height
        col_name = f'Horizontal Wind Speed (m/s) at {height}m'

        if col_name not in lidar_data.columns:
            print(f"Column '{col_name}' not found in the dataset!")
            continue  # Skip to next height if the column is not found

        # Apply the interpolation function (replace NaN values with linear interpolation)
        lidar_data[col_name] = interpolate_with_threshold(lidar_data[col_name])

        print(f"Processing data for {col_name}...")

        # Create a mask to identify non-NaN values
        mask = ~np.isnan(lidar_data[col_name])

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

        # Compute acceleration in the time domain
        spectral_accel_time = compute_spectral_acceleration_time(valid_data, fs)

        # Now, restore NaNs back to their original positions in the time-domain result
        spectral_accel_time_full = np.full_like(lidar_data[col_name], np.nan)

        # Insert the spectral acceleration values into the valid positions (where mask is True)
        spectral_accel_time_full[mask] = spectral_accel_time[:len(valid_data)]

        # Finite difference
        finite_accel = central_diff_acceleration(lidar_data[col_name], 11)

        # Create 3-minute windows for the acceleration time series
        window_size = int(180 / 11)  # Number of samples per 3-minute window
        percentiles_values = compute_percentiles(spectral_accel_time_full, window_size, percentiles=[90])
        percentiles_finite = compute_percentiles(finite_accel, window_size, percentiles=[90])

        # P90 accelerations
        p90_values = percentiles_values[90]
        p90_values1 = percentiles_finite[90]

        # Stats on P90 acceleration
        mean_p90 = np.mean(p90_values)
        std_p90 = np.std(p90_values)

        # PLOTTING

        # Spectrum
        plt.figure(figsize=(10, 6))
        plt.loglog(freqs_u, freqs_u * psd_u, label=f'Raw Spectrum {height}m')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power Spectral Density')
        plt.legend()
        plt.grid(True)

        # Smoothed spectrum
        plt.figure(figsize=(10, 6))
        plt.loglog(freqs_u, freqs_u * psd_u, label=f'Raw Spectrum {height}m', alpha=0.5)
        plt.loglog(smoothed_freqs, smoothed_freqs * smoothed_spectrum, label=f'Smoothed Spectrum {height}m', linewidth=2)
        plt.legend()
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(r'fS(f) [$m^2/s^2$]')

        # Acceleration time domain
        plt.figure(figsize=(10, 6))
        plt.plot(timestamp / 3600, spectral_accel_time_full, label=f'Spectral Acceleration {height}m', color='red', alpha=0.7)
        plt.plot(timestamp / 3600, finite_accel, label=f'Finite Acceleration {height}m', color='green', alpha=0.7)
        plt.xlabel('Time [hr]')
        plt.ylabel(r'Acceleration [$m/s^2$]')
        plt.legend()
        plt.grid(True)

        # 90th Percentile Histogram (Acceleration on x-axis)
        plt.figure(figsize=(10, 6))
        plt.hist(p90_values, bins=40, color='green', alpha=0.7, edgecolor='black', label=f'Spectral Accel {height}m')
        plt.hist(p90_values1, bins=40, color='blue', alpha=0.7, edgecolor='black', label=f'Finite Accel {height}m')
        plt.xlabel('Acceleration (m/s²)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of 90th Percentile Acceleration {height}m')
        plt.legend()
        plt.grid(True)
        print("loop 1")
    plt.show()

# Run the main function
main()
