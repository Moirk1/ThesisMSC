import numpy as np
#FFT get spectrum from time series
def compute_psd(time_series, fs):
    # Perform FFT and calculate power spectral density (PSD)
    N = len(time_series)
    freqs = np.fft.fftfreq(N, 1/fs) #find frequency bins(Number of samples, time step)
    fft_vals = np.fft.fft(time_series) #fft on time_series
    psd = (np.abs(fft_vals)**2) / (N*fs) #find  power spectral density
    return freqs[:N//2], psd[:N//2]  # Return only positive frequencies and corresponding PSD since FFT is symmetric for real time series

#Get spectral acceleration from spectrum
def compute_spectral_acceleration(freqs, psd):
    """Compute the spectral acceleration from the power spectral density (PSD)."""
    # Spectral acceleration is calculated as (2 * pi * f)^2 * PSD
    spectral_accel = (2 * np.pi * freqs)**2 * psd
    return spectral_accel

#Use IFT to get spectral acceleration in time domain
def compute_spectral_acceleration_time(fft_vals, freqs):
    """Compute spectral acceleration in the time domain using inverse FFT."""
    spectral_accel_fft = (2 * np.pi * freqs) ** 2 * fft_vals  # Modify FFT spectrum
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