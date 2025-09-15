"""Functions to finding landing events"""
import numpy as np

def get_end_of_landing_phase(velocity_series) -> int:
    """Function to get the end of the landing phase of a jump.

    Args:
        velocity_series (array): Velocity time series of the landing phase of a jump

    Returns:
        int: Frame associated with the end of the landing phase
    """
    velocities_greater_than_zero = np.where(velocity_series >= 0)[0]
    if len(velocities_greater_than_zero) > 0:
        return velocities_greater_than_zero[0]
    else:
        return -1
