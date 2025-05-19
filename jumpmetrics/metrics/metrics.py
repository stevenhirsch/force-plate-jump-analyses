"""Functions to compute CMJ metrics"""
import logging
import numpy as np
from numpy.typing import NDArray
from jumpmetrics.signal_processing.numerical import (
    compute_derivative
)
NOT_FOUND = -10 ** 2

def get_bodyweight(force_series, sampling_frequency: float = 2000, sec: float = 0.25) -> float:
    """Function to compute someone's bodyweight based on their static stance. The first n frames
    are assumed to be "static

    Args:
        force_series (NDArray): Force series of jump
        sampling_frequency (float, optional): Sampling frequency of force plate. Defaults to 2000.
        sec (float, optional): Number of seconds to weigh someone. Defaults to 0.25.

    Returns:
        float: Bodyweight in newtons
    """
    n = int(sampling_frequency * sec)
    average_of_first_n_frames = force_series[0:n].mean()
    return average_of_first_n_frames

def compute_rfd(
    force_trace, window_start: int, window_end: int, sampling_frequency: float, method: str = 'average'
) -> float:
    """Function to compute rate of force development (RFD) during a jump between specific windows and using 
    various methods.

    Args:
        force_trace (array): Force series
        window_start (int): Start of the window for computing RFD
        window_end (int): End of the window for computing RFD
        sampling_frequency (float): Sampling frequency of the force plate
        method (str, optional): Method for computing RFD. Defaults to 'average'.

    Returns:
        float: RFD in Newtons per second
    """
    # if window_end is None:
    if window_end == NOT_FOUND:
        logging.warning('End of window for RFD not found, returning np.nan')
        return np.nan
    # if window_start is None:
    if window_start == NOT_FOUND:
        logging.warning('Start of window for RFD not found, returning np.nan')
        return np.nan
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
            rfd = np.nanmax(rfd_between_events)
        else:
            rfd = np.nanmean(rfd_between_events)
    else:
        logging.error("%s is not a valid method. Please select one of: %s", method, valid_methods)
        rfd = None

    return rfd

def compute_jump_height_from_takeoff_velocity(takeoff_velocity: float) -> float:
    """Function to compute jump height based on someone's takeoff velocity

    Args:
        takeoff_velocity (float): Takeoff velocity of a jump

    Returns:
        float: Jump height in meters
    """
    jump_height = (takeoff_velocity ** 2) / (2 * 9.81)
    return jump_height

def compute_jump_height_from_velocity_series(velocity_series: NDArray[np.float64]) -> float:
    """Function to compute jump height from the last value of a velocity series
    before takeoff.

    Args:
        velocity_series (np.ndarray): Array of movement velocities

    Returns:
        float: Jump height in meters
    """
    takeoff_velocity = velocity_series[-1]  # take the last frame
    jump_height = compute_jump_height_from_takeoff_velocity(takeoff_velocity)
    return jump_height

def compute_jump_height_from_net_vertical_impulse(net_vertical_impulse: float, body_mass_kg: float) -> float:
    """Function to compute the jump height from the net vertical impulse of a CMJ

    Args:
        net_vertical_impulse (float): Net vertical impulse of a jump
        body_mass_kg (float): Participants' body mass (in KG)

    Returns:
        float: Jump height in meters
    """
    impulse_takeoff_velocity = net_vertical_impulse / body_mass_kg  # impulse = momentum = mass * velocity
    jump_height = compute_jump_height_from_takeoff_velocity(takeoff_velocity=impulse_takeoff_velocity)
    return jump_height

def compute_average_force_between_events(force_trace, window_start: int, window_end: int) -> float:
    """Function to compute the average force between two events

    Args:
        force_trace (array): Force series
        window_start (int): Frame corresponding to the start of when to compute average force
        window_end (int): Frame corresponding to the end of when to compute average force

    Returns:
        float: Average force between the events, in Newtons
    """
    if window_start == NOT_FOUND:
        logging.warning("window_start is -1, returning np.nan")
        return np.nan
    if window_end == NOT_FOUND:
        logging.warning('window_end is -1, returning np.nan')
        return np.nan
    if window_end <= window_start and window_end > 0:
        logging.warning(
            """window_end occurred before or at the same time as window_start,
            so average force between events is invalid. Returning np.nan"""
        )
        return np.nan
    average_force = force_trace[window_start:window_end].mean()
    return average_force

def compute_jump_height_from_total_flight_time(
    flight_time: float
) -> float:
    """Function to compute jump height from the total flight time

    Args:
        flight_time (float): Total flight time (takeoff to landing) in seconds

    Returns:
        float: Vertical jump height (in meters)
    """
    accel_gravity = 9.81
    time_to_peak_height = flight_time / 2
    jump_height = 0.5 * accel_gravity * time_to_peak_height ** 2
    return jump_height


def compute_jump_height_from_flight_time_events(
    takeoff_frame: int, landing_frame: int, sampling_frequency: float
) -> float:
    """Function to compete jump height from flight time events

    Args:
        takeoff_frame (int): Frame corresponding to takeoff
        landing_frame (int): Frame corresponding to landing
        sampling_frequency (float): Sampling frequency (frames/samples per second)

    Returns:
        float: Jump height (in meters)
    """
    flight_time = (landing_frame - takeoff_frame) / sampling_frequency
    jump_height = compute_jump_height_from_total_flight_time(
        flight_time=flight_time
    )
    return jump_height
