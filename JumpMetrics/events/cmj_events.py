import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks  # Handy function for finding peaks from the SciPy library


def get_start_of_jump(series, threshold=0.05):
    frame = np.where(series < -threshold)[0][0]
    return frame

def get_start_of_jump_based_on_stance_sd(series, n=500):
    average_weight = series.iloc[0:n].mean()
    sd_weight = series.iloc[0:n].std()
    series_average_diff = series - average_weight
    frame = np.where(series_average_diff < -sd_weight*5)[0][0]  # 5 standard deviations below the SD during stance
    return frame

def get_start_of_jump_based_on_accel(accel_series, threshold=-1):
    jerk = accel_series.diff()
    frame = np.where(jerk <= threshold)[0][0]
    return frame

def find_unweighting_start(force_data, sample_rate, quiet_period=1, threshold_factor=5, window_size=0.2, duration_check=0.1):
    # Convert window size from seconds to samples
    window_samples = int(window_size * sample_rate)
    
    # Smooth the force data using Savitzky-Golay filter
    smoothed_force = savgol_filter(force_data, window_samples, 3)
    
    # Calculate mean and standard deviation of the quiet period
    quiet_samples = int(quiet_period * sample_rate)
    quiet_force = force_data[:quiet_samples]
    mean_force = np.mean(quiet_force)
    std_force = np.std(quiet_force)
    
    # Calculate the threshold
    threshold = mean_force - threshold_factor * std_force
    
    # Convert duration_check from seconds to samples
    duration_samples = int(duration_check * sample_rate)
    
    # Find the first point where smoothed force drops below threshold
    unweighting_start = None
    for i in range(quiet_samples, len(smoothed_force) - duration_samples):
        if smoothed_force[i] < threshold:
            # Check if force stays below threshold for the specified duration
            if np.all(smoothed_force[i:i+duration_samples] < threshold):
                unweighting_start = i
                break
    
    return unweighting_start

def get_start_of_concentric_phase(velocity_series, frame_of_eccentric_start):
    # frame = np.where(velocity_series[frame_of_eccentric_start:] > 0)[0][0]
    min_velocity_index = np.argmin(velocity_series)
    frame = np.where(velocity_series[min_velocity_index:] > 0)[0][0]
    return frame + min_velocity_index

def get_start_of_braking_phase_using_velocity(velocity_series):
    return np.argmin(velocity_series)

def get_start_of_propulsive_phase_using_displacement(displacement_series):
    return np.argmin(displacement_series)

def get_peak_force_event(force_series, start_of_propulsive_phase):
    peaks, _ = find_peaks(
        force_series.iloc[start_of_propulsive_phase:], prominence=50
    )
    if len(peaks) == 0:
        peak_force_frame = force_series.iloc[start_of_propulsive_phase:].idxmax()
    else:
        peak_force_frame = peaks[0] + start_of_propulsive_phase
    return peak_force_frame
