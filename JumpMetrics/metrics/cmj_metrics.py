import numpy as np
import logging  # for logging errors
from JumpMetrics.signal_processing.numerical import (
    compute_derivative
)


# get bodyweight (assumes first n frames are "static" and represent someone's bodyweight in Newtons)
def get_bodyweight(force_series, n=500):
    average_of_first_n_frames = force_series.iloc[0:n].mean()
    return average_of_first_n_frames

def compute_rfd(force_trace, window_start, window_end, sampling_frequency, method='average'):
    if window_end <= window_start:
        logging.warning("window_end is before or equal to window_start, so RFD will be invalid. Returning np.nan")
        return np.nan
    valid_methods = ['average', 'instantaneous', 'peak']
    if method == 'average':
        force_at_start = force_trace[window_start]
        force_at_end = force_trace[window_end]
        num_frames_between_points = window_end - window_start
        time_between_points = num_frames_between_points / sampling_frequency
        rfd = (force_at_end - force_at_start) / time_between_points
    elif method == 'instantaneous' or method == 'peak':
        force_trace_derivative = compute_derivative(force_trace, t=1/sampling_frequency, d=1)
        rfd_between_events = force_trace_derivative[window_start:window_end]
        if method == 'peak':
            rfd = np.max(rfd_between_events)
        else:
            rfd = np.mean(rfd_between_events)
    else:
        logging.error(f"{method} is not a valid method. Please select one of: {valid_methods}")
        rfd = None

    return rfd

def compute_jump_height_from_takeoff_velocity(takeoff_velocity):
    jump_height = (takeoff_velocity ** 2) / (2 * 9.81)
    return jump_height

def compute_jump_height_from_velocity_series(velocity_series):
    takeoff_velocity = velocity_series[-1]  # take the last frame
    jump_height = compute_jump_height_from_takeoff_velocity(takeoff_velocity)
    return jump_height

def compute_jump_height_from_net_vertical_impulse(net_vertical_impulse, body_mass_kg):
    impulse_takeoff_velocity = net_vertical_impulse / body_mass_kg  # impulse = momentum = mass * velocity
    jump_height = compute_jump_height_from_takeoff_velocity(takeoff_velocity=impulse_takeoff_velocity)
    return jump_height

def compute_average_force_between_events(force_trace, window_start, window_end):
    if window_end <= window_start and window_end > 0:
        logging.warning(f"window_end occurred before or at the same time as window_start, so average force between events is invalid. Returning np.nan")
        return np.nan
    average_force = force_trace.iloc[window_start:window_end].mean()
    return average_force