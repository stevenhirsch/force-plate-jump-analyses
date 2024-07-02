from scipy import signal
import numpy as np


def butterworth_filter(
    arr: np.ndarray, cutoff_frequency: float, fps: float, padding: int, order: int= 4
) -> np.ndarray:
    """Function to apply a low-pass butterworth filter to a signal

    Args:
        arr (np.ndarray): Input array
        cutoff_frequency (float): Lowpass filter cutoff
        fps (float): Frames per second of the trial
        padding (int): Amount of frames to pad the signal
        order (int, optional): Order of the filter. Defaults to 4.

    Returns:
        np.ndarray: Filtered signal
    """
    w = cutoff_frequency / (fps / 2)  # Normalize the frequency
    b, a = signal.butter(order, w, 'low')
    if len(arr) - 1 < fps:  # i.e., the length of the arr is less than 1 second
        output = signal.filtfilt(b, a, arr, method='gust')
    else:
        output = signal.filtfilt(b, a, arr, padlen=padding, method='pad')
    return output