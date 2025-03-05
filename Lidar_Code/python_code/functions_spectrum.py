import numpy as np
#FFT get spectrum from time series
def compute_psd(time_series, fs):
    # Perform FFT and calculate power spectral density (PSD)
    N = len(time_series)
    freqs = np.fft.fftfreq(N, 1/fs) #find frequency bins(Number of samples, time step)
    fft_vals = np.fft.fft(time_series) #fft on time_series
    psd = (np.abs(fft_vals)**2) / (N*fs) #find  power spectral density
    return freqs[:N//2], psd[:N//2]  # Return only positive frequencies and corresponding PSD since FFT is symmetric for real time series


#Use IFT to get spectral acceleration in time domain
def compute_spectral_acceleration_time(time_series, fs):
    """Compute spectral acceleration in the time domain using inverse FFT."""
    N = len(time_series)
    freqs = np.fft.fftfreq(N, 1/fs) #find frequency bins(Number of samples, time step)
    fft_vals = np.fft.fft(time_series) #fft on time_series
    psd = (np.abs(fft_vals)**2) / (N*fs) #find  power spectral density
    spectral_accel_fft = (2 * np.pi * freqs)*1j * fft_vals # Modify FFT spectrum
    spectral_accel_time = np.fft.ifft(spectral_accel_fft).real  # Inverse FFT back to time
    return spectral_accel_time

#Spectrum checker for unsmoothed time series (excludes initial value)
def speccheck_unsmooth(title, timeseries, freqs, psd): # Oi speci are you for real?
    N = len(timeseries)
    sigma_t = np.std(timeseries) #sigma of time series
    area = np.trapz(psd[1:N//2], freqs[1:N//2]) #area under spectrums old np.trapz(psd, freqs)
    sigma_s = np.sqrt(2*area)
    print(f"SPECCHECK! on {title}")
    print(f"Sigma of time series: {sigma_t}")
    print(f"Sigma from Spectrum: {sigma_s}")
    print("SPECCHECK done")
    return

#Spectrum checker for smoothed time series (does not exclude initial value)
def speccheck_smooth(title, timeseries, freqs, psd): # Oi speci are you for real?
    N = len(timeseries)
    sigma_t = np.std(timeseries) #sigma of time series
    area = np.trapz(psd, freqs) #area under spectrums old np.trapz(psd, freqs)
    sigma_s = np.sqrt(2*area)
    print(f"SPECCHECK! on {title}")
    print(f"Sigma of time series: {sigma_t}")
    print(f"Sigma from Spectrum: {sigma_s}")
    print("SPECCHECK done")
    return

#Logarithmic smoothing function
def smooth_criminal(spectrum, f, n_per_decade):
    f_min = np.log10(f[1])  #define min of log bins (This can probably be done better but 1 since I've shifted hov)
    f_max = np.log10(f[-1])

    # Number of decades in the frequency range
    num_decades = f_max-f_min
    #create log bins
    log_bins = np.logspace(f_min, f_max, int(num_decades * n_per_decade))  #generate numbers spaced evenly on a logarithmic scale
    #initialise arrays
    smoothed_freq = []
    smoothed_spectrum = []
    # Step 3: Perform averaging within each logarithmic bin
    for i in range(len(log_bins)-1): #must have -1 as log bins contains end point of bins
        bin_start = log_bins[i] 
        bin_end = log_bins[i+1]
    # Find indices of spectrum values that fall into this bin
        in_bin = np.where((f>=bin_start)&(f<bin_end))[0] #0 makes it array accessible
        # Average the spectrum values within the bin
        if len(in_bin) > 0:
            # Average the spectrum values within the bin
            smoothed_spectrum.append(np.mean(spectrum[in_bin]))
            smoothed_freq.append(np.mean(f[in_bin]))

# Convert lists to numpy arrays for easier use
    smoothed_spectrum = np.array(smoothed_spectrum)
    smoothed_freq = np.array(smoothed_freq)

    return smoothed_freq, smoothed_spectrum

# Function to handle interpolation with missing data threshold (Works on everything except 0 value)
def interpolate_with_threshold(data, threshold=2):
    data_copy = data.copy()  # Work on a copy to avoid modifying the original
    
    # Track the positions of NaN values
    nan_indices = data_copy.index[data_copy.isna()]
    
    # List to store indices to interpolate
    interp_indices = []
    
    # Find groups of consecutive NaNs
    i = 0
    while i < len(nan_indices):
        consecutive_nans = [nan_indices[i]]
        j = i + 1
        while j < len(nan_indices) and nan_indices[j] - nan_indices[j - 1] == 1:
            consecutive_nans.append(nan_indices[j])
            j += 1
        if len(consecutive_nans) <= threshold:
            interp_indices.extend(consecutive_nans)  # Add indices for interpolation
        i = j  # Move to the next set of NaNs
    
    # Perform interpolation on the full dataset
    interpolated_series = data_copy.interpolate(method='linear')

    # Only apply changes to the selected NaN indices
    for idx in interp_indices:
        data_copy.iloc[idx] = interpolated_series.iloc[idx]
    
    return data_copy

#Find percentiles
def compute_percentiles(time_series, window_size, percentiles=[90]):
    """Compute the desired percentiles (P90) for each window of data, ignoring NaNs."""
    percentiles_values = {p: [] for p in percentiles}
    
    print(f"Length of time_series: {len(time_series)}")
    print(f"Window size: {window_size}")

    # Iterate through the time series with the given window size
    for start in range(0, len(time_series), window_size):
        end = start + window_size
        # Get the window of data
        window_data = time_series[start:end]

        print(f"Start index: {start}, End index: {end}, Window size: {len(window_data)}")
        
        # Remove NaNs from the window
        window_data = window_data[~np.isnan(window_data)]  # Ignore NaNs in this window
        
        # If the window has enough valid data, compute percentiles
        if len(window_data) > 0:
            for p in percentiles:
                percentiles_values[p].append(np.percentile(window_data, p))
        else:
            print(f"Skipping window from index {start} to {end} due to lack of valid data.")
    
    # Convert the results to numpy arrays for easier handling
    for p in percentiles_values:
        percentiles_values[p] = np.array(percentiles_values[p])

    return percentiles_values

#Finite difference acceleration
def central_diff_acceleration(velocity, delta_t):
    velocity = np.array(velocity)
    # Create an empty array to store acceleration values
    acceleration = np.full_like(velocity, np.nan)  # Initialize with NaNs
    # Iterate over the time series (starting from the second element and ending before the last element)
    for t in range(1, len(velocity) - 1):
        # Apply the central difference formula
        acceleration[t] = (velocity[t + 1] - velocity[t - 1]) / (2 * delta_t)
    return acceleration
