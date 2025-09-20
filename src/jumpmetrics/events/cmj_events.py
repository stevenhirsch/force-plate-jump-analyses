"""Functions for CMJ events"""
import logging
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks  # Handy function for finding peaks from the SciPy library

NOT_FOUND = -10 ** 2

def find_unweighting_start(
    force_data, sample_rate: float, quiet_period: float = 1.,
    threshold_factor: float = 5., window_size: float = 0.2, duration_check: float = 0.1
) -> int:
    """Function to find the start of the unweighting phase

    Args:
        force_data (array): Force series

        sample_rate (float): Sampling rate of the force plate

        quiet_period (float, optional): This is the duration (in seconds) at the beginning of the
        data that you consider to be the "quiet stance" period. During this time, the participant should
        be standing relatively still. The default is set to 1 second, but you might adjust this based on
        your experimental protocol. Defaults to 1 second.

        threshold_factor (float, optional): This is the number of standard deviations below the mean force
        that will be used to determine the start of unweighting. The default is 5, which means the
        algorithm will look for force values that drop below (mean force - 5 * standard deviation). You
        might adjust this if you find the algorithm is triggering too early or too late. Defaults to 5
        standard deviations.

        window_size (float, optional): This is the size of the window (in seconds) used for
        the Savitzky-Golay filter, which smooths the data. The default is 0.2 seconds. A larger
        window will result in more smoothing but might also blur important features of the force curve.
        Defaults to 0.2.

        duration_check (float, optional): Number of seconds to check if the person is unweighting.
        Used to ignore false positives. Defaults to 0.1 seconds.

    Returns:
        int: Frame number corresponding to the start of the unweighting phase
    """
    # Input validation
    if len(force_data) < sample_rate * quiet_period:
        raise ValueError(f"Force data too short ({len(force_data)} samples) for quiet period ({quiet_period}s at {sample_rate}Hz)")

    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")

    if quiet_period <= 0:
        raise ValueError(f"Quiet period must be positive, got {quiet_period}")

    if duration_check <= 0:
        raise ValueError(f"Duration check must be positive, got {duration_check}")

    if window_size <= 0:
        raise ValueError(f"Window size must be positive, got {window_size}")
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
    if unweighting_start is None:
        logging.info(f"No unweighting detected with threshold_factor={threshold_factor}. "
                    f"Try lowering threshold_factor or check if this is actually a countermovement jump.")
        return NOT_FOUND
    else:
        logging.debug(f"Unweighting detected at frame {unweighting_start} ({unweighting_start/sample_rate:.3f}s)")
        return unweighting_start

def get_start_of_braking_phase_using_velocity(velocity_series, start_of_unweighting_phase) -> int:
    """Function to find the start of the braking phase using the velocity waveform

    Args:
        velocity_series (array): Velocity waveform of the jump
        start_of_unweighting_phase: Frame where unweighting started

    Returns:
        int: Frame associated with the start of the braking phase
    """
    if start_of_unweighting_phase >= 0:
        return int(start_of_unweighting_phase + np.argmin(velocity_series[start_of_unweighting_phase:]))
    else:
        logging.warning('Not using unweighting phase to help find braking phase')
        return int(np.argmin(velocity_series))

def get_start_of_propulsive_phase_using_displacement(displacement_series, start_of_braking_phase) -> int:
    """Function to find the start of the propulsive phase using the displacement data

    Args:
        displacement_series (array): Displacement array of the jump

    Returns:
        int: Frame associated with the start of the propulsive phase
    """
    if start_of_braking_phase is not None:
        return int(start_of_braking_phase + np.argmin(displacement_series[start_of_braking_phase:]))
    else:
        logging.warning('Not using braking phase to help find propulsive phase')
        return int(np.argmin(displacement_series))

def get_peak_force_event(force_series, start_of_propulsive_phase: int) -> int:
    """Find the frame associated with the "peak" of a waveform. Note that this looks for a peak after
    the start of the propulsive phase, which requires a certain prominence to be detected. Please see
    scipy's find_peaks() documentation for more information about prominence.

    Args:
        force_series (array): Force waveform
        start_of_propulsive_phase (int): Frame associated with the start of the propulsive phase

    Returns:
        int: Frame corresponding to the peak force
    """
    if start_of_propulsive_phase is None or start_of_propulsive_phase < 0:
        logging.warning('Invalid propulsive phase start, using global maximum for peak force detection')
        return int(np.argmax(force_series))

    peaks, _ = find_peaks(
        force_series[start_of_propulsive_phase:], prominence=50
    )
    if len(peaks) == 0:
        peak_force_frame = int(np.argmax(force_series[start_of_propulsive_phase:]) + start_of_propulsive_phase)
    else:
        peak_force_frame = int(peaks[0] + start_of_propulsive_phase)
    return peak_force_frame
