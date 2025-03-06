import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions_spectrum import *  # Assuming the functions are imported from this file
import seaborn as sns  # Make sure seaborn is installed

def main():
    # Initial setup
    lidar_data = pd.read_csv('modified_lidar_data21_2025-02-13_113639.csv')  # Import data files
    timestamp = lidar_data['Timestamp (s)'] - 808012801  # Adjust timestamp as required
    fs = 1 / 11  # Sample frequency in Hz
    heights = [300, 250, 200, 150, 100]  # List of heights to process
    p90_accel_dict = {}  # Dictionary to hold P90 accelerations for each height
    nan_percentage_before = []  # List to store NaN percentages before interpolation
    nan_percentage_after = []   # List to store NaN percentages after interpolation


    for height in heights:
        # Construct the column name based on height
        col_name = f'Horizontal Wind Speed (m/s) at {height}m'

        if col_name not in lidar_data.columns:
            print(f"Column '{col_name}' not found in the dataset!")
            continue  # Skip to next height if the column is not found
        
        # Calculate the percentage of NaN values before interpolation
        nan_before = lidar_data[col_name].isna().sum()  # Count NaN values
        total_values = len(lidar_data[col_name])  # Total number of values
        nan_percentage_before.append((height, (nan_before / total_values) * 100))  # Calculate percentage
        # Apply the interpolation function (replace NaN values with linear interpolation)
        lidar_data[col_name] = interpolate_with_threshold(lidar_data[col_name])

        # Calculate the percentage of NaN values after interpolation
        nan_after = lidar_data[col_name].isna().sum()  # Count NaN values after interpolation
        nan_percentage_after.append((height, (nan_after / total_values) * 100))  # Calculate percentage


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
        window_size3 = int(180 / 11)  # Number of samples per 3-minute window
        window_size10 = int(600/11)
        percentiles_values = compute_percentiles(spectral_accel_time_full, window_size3, percentiles=[90])
        percentiles_10min = compute_percentiles(spectral_accel_time_full, window_size10, percentiles=[90])

        # P90 accelerations
        p90_values = percentiles_values[90]
        p90_values1 = percentiles_10min[90]

         # Add P90 values for this height to the dictionary
        p90_accel_dict[height] = p90_values  # Store the P90 accelerations by height

        # Stats on P90 acceleration
        mean_p90 = np.mean(p90_values)
        std_p90 = np.std(p90_values)

        # PLOTTING
        """
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
        plt.hist(p90_values, bins=40, color='green', alpha=0.7, edgecolor='black', label=f'3 minute base period {height}m')
        plt.hist(p90_values1, bins=40, color='blue', alpha=0.7, edgecolor='black', label=f'10 minute base period {height}m')
        plt.xlabel('P90 accelerations (m/s²)', fontsize = 14)
        plt.ylabel('Frequency', fontsize = 14)
        #plt.title(f'Histogram of 90th Percentile Acceleration {height}m')
        plt.legend(fontsize = 12)
        plt.grid(True)

        """
    # Create a DataFrame to display the comparison in a table
    nan_comparison_df = pd.DataFrame({
        'Height (m)': [height for height, _ in nan_percentage_before],
        'NaN Percentage Before (%)': [nan_before for _, nan_before in nan_percentage_before],
        'NaN Percentage After (%)': [nan_after for _, nan_after in nan_percentage_after],
    })

    # Display the table
    print(nan_comparison_df)
    
    # Heatmap plotting
    # For the heatmap, we'll use the P90 accelerations and heights
    # Create a list of P90 accelerations and their respective heights
    heatmap_data = []
    for height, p90_vals in p90_accel_dict.items():
        # Bin P90 values into groups, here we use 0.5 m/s² intervals as an example
        bins = np.arange(-0.15, np.max(p90_vals) + 0.15, 0.02)
        digitized = np.digitize(p90_vals, bins)  # Digitize the P90 values into bins

        # Collect frequency for each height and binned P90 value
        for bin_idx in digitized:
            heatmap_data.append([height, bins[bin_idx - 1], 1])  # bin_idx - 1 to get the actual bin start value

    # Convert the data into a DataFrame
    heatmap_df = pd.DataFrame(heatmap_data, columns=['Height (m)', 'P90 Acceleration (m/s²)', 'Frequency'])

    # Pivot the DataFrame for heatmap
    # Swap height and P90 acceleration for axis flipping
    heatmap_df_pivot = heatmap_df.pivot_table(index='P90 Acceleration (m/s²)', columns='Height (m)', values='Frequency', aggfunc='sum', fill_value=0)

    # Format P90 acceleration to 3 decimal places
    heatmap_df_pivot.index = heatmap_df_pivot.index.round(3)

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_df_pivot, cmap='viridis', annot=True, fmt='d', linewidths=0.5, cbar_kws={'label': 'Frequency'})
    plt.xlabel('Height (m)')
    plt.ylabel('P90 Acceleration (m/s²)')
    plt.show()

    

# Run the main function
main()
