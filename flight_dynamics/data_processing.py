import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    flight_data = pd.read_csv('2024_week_17_thu_03\kite.csv')
    timestamp = flight_data['epoch_ms']-flight_data['epoch_ms'][0] #Changing start time to 0
  
   
       # Get the tether length data
    tether_length = flight_data['tether_length'].values
    
    # Step 1: Apply a smoothing function (e.g., moving average or polynomial fit)
    window_size = 200  # Define the window size for the moving average
    smoothed_tether_length = np.convolve(tether_length, np.ones(window_size)/window_size, mode='same')
    
    # Step 2: Compute the slope (derivative) of the smoothed signal
    slope = np.diff(smoothed_tether_length)
    
    # Step 3: Identify segments with a negative slope and set them to NaN
    tether_length_with_nans = tether_length.copy()
    tether_length_with_nans[1:][slope < 0] = np.nan  # Skip the first element as np.diff reduces length by 1

    # Step 4: Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(timestamp, tether_length_with_nans, label='Tether Length with Negative Slopes Removed')
    plt.plot(timestamp, smoothed_tether_length, label='Smoothed Trend', linestyle='--', color='red')
    plt.xlabel('Time [s]')
    plt.ylabel('Tether Length [m]')
    plt.grid(True)
    plt.legend()
    plt.title('Tether Length Over Time (Negative Slopes Removed, Smoothed Trend)')
    plt.show()
   
main()