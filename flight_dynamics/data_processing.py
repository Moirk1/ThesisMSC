import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Load the flight data
    flight_data = pd.read_csv('2024_week_17_thu_03\\kite.csv')
    timestamp1 = flight_data['epoch_ms'] - flight_data['epoch_ms'][0] 
    # Make a copy of the original data for comparison 
    modified_data = flight_data.copy()  # Create a copy of the original data
    timestamp = modified_data['epoch_ms'] - modified_data['epoch_ms'][0]  # Adjust timestamp to start at 0
  
    # Get the tether length data
    tether_length = modified_data['tether_length'].values

    # Step 1: Manually takeoff and landing periods 
    takeoff_end_index = 2000  
    landing_start_index = 83600  

    # Set NaNs from the start to takeoff_end_index (this can be your takeoff period)
    modified_data.iloc[:takeoff_end_index + 1, 1:] = np.nan  # Skip the first column (timestamp)

    # Set NaNs from landing_start_index to the end of the dataset (this can be your landing period)
    modified_data.iloc[landing_start_index:, 1:] = np.nan  # Skip the first column (timestamp)

    # Step 2: Apply a smoothing function 
    window_size = 200  # Define the window size for the moving average
    smoothed_tether_length = np.convolve(tether_length, np.ones(window_size)/window_size, mode='same')
    
    # Step 3: Compute the slope (derivative) of the smoothed signal
    slope = np.diff(smoothed_tether_length)
    
    # Step 4: Identify segments with a negative slope and set them to NaN
    tether_length_with_nans = tether_length.copy()
    tether_length_with_nans[1:][slope < 0] = np.nan  # Skip the first element as np.diff reduces length by 1
    # Step 5: Propagate NaNs from the 'tether_length_with_nans' column to all other columns (except timestamp)
    for column in flight_data.columns:
        if column != 'epoch_ms':  # Don't apply NaN to the 'epoch_ms' (timestamp) column
            modified_data[column] = np.where(np.isnan(tether_length_with_nans), np.nan, modified_data[column])
    

    # Extract the original filename without extension
    original_filename = os.path.splitext('2024_week_17_thu_03\\kite.csv')[0]
    # Create a new filename for the modified dataset
    new_filename = f"{original_filename}_power_production.csv"
    # Save the modified dataset
    modified_data.to_csv(new_filename, index=False)
    
    # Step 5: Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(timestamp1, flight_data['tether_length'], label = 'Initial dataset', color = 'red', linestyle = '--')
    plt.plot(timestamp, tether_length_with_nans, label='Power Production Phase', color='blue')
    plt.xlabel('Time [s]')
    plt.ylabel('Tether Length [m]')
    plt.grid(True)
    plt.legend()
    #plt.title('Tether Length with NaNs During Takeoff, Landing, and Negative Slopes')
    #plt.savefig('tether_length_plot.pdf', format='pdf')
    plt.show()

main()
