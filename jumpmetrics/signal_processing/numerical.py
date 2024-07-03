"""Functions for numerical operations"""
import numpy as np
from scipy.integrate import simps  # Simpson's method for computing integral
import pandas as pd


def integrate_area(time, signal, method='trapezoidal') -> np.ndarray:
    """Functino to compute the area under a curve using the integral

    Args:
        time (array): Array of timestamps
        signal (array): Array of the signal to integrate over time
        method (str, optional): Integration method. Defaults to 'trapezoidal'.

    Raises:
        ValueError: Raised if the method is not trapezoidal or Simpson's

    Returns:
        np.ndarray: Area under the curve
    """
    # Ensure time and signal are numpy arrays
    time = np.asarray(time)
    signal = np.asarray(signal)

    if method == 'trapezoidal':
        # Compute the area using the trapezoidal rule
        area = np.trapz(signal, time)
    elif method == 'simpsons':
        # Compute the area using Simpson's rule
        area = simps(signal, time)
    else:
        raise ValueError("Method must be 'trapezoidal' or 'simpsons'.")

    return area

def get_finite_difference_coefficients(d: int, order_accuracy: int) -> np.ndarray:
    '''
    Compute the coefficients for the finite difference equation corresponding to the derivative, d, to a 
    specified order of accuracy.
    

    Parameters
    -----------------
    d: int
        The derivative in which we are approximating (e.g. 1st, 2nd, 3rd, etc.).
    
    order_accuracy: int
        The order of accuracy in which the finite difference equation will approximate to. This must be a
        multiple of 2 for the symmetric difference.

    Returns
    -----------------

    coefficients: numpy.ndarray
        The coefficients used in the finite difference equation to approximate the derivative.
    '''
    # N number of coefficients
    # The if statement below is used to ensure that the correct order accuracy is specified and to raise an
    # appropriate error if not specified.
    if order_accuracy % 2 == 0:
        if d % 2 == 0:
            n_pre = d + order_accuracy - 1
        else:
            n_pre = d + order_accuracy
    else:
        raise ValueError('Order of Accuracy must be a multiple of 2 for the symmetric difference')

    # n represents the number of coefficients pre and post the frame of interest. This is used to create a
    # vector "master_row" below.
    n = int((n_pre-1)/2)

    master_row = np.arange(n * -1, n + 1)
    # Creating an NxN matrix as a placeholder (note that the matrix indices shown in (6) start at 0 and end
    # at N-1, so its dimension is NxN).
    matrix = np.zeros([len(master_row), len(master_row)])

    # This loop raises each element of master_row to the exponent i. As noted in the previous text, there is
    # a pattern in which the center of the rows are always =0, and each number to the right corresponds to +1,
    # +2, +3... etc. and to the left corresponds to -1, -2, -3,... etc. Calculating the matrix of the
    # constants from the Taylor Expansion is therefore as simple as raising the elements in master_row to the
    # exponent of the row number (i.e. the index of the for loop).
    for i in range(len(master_row)):
        matrix[i] = master_row ** i

    # Create a dummy vector named 'vec' to ultimately hold the right-hand side of our system of equations.
    vec = np.zeros(len(master_row))
    np.put(vec, d, np.math.factorial(d))
    # As shown above, our coefficients are then calculated by solving the system of linear equations in the
    # standard manner.
    coefficients = np.matmul(np.linalg.inv(matrix), vec)


    # Uncommenting the code below will add a tolerance to force what "should be" zero values
    # to return as 0.
    # If uncommenting, you can adjust the tolerance to whatever you feel may be more appropriate.

    # tolerance = 1e-10
    # coefficients[abs(coefficients.real) < tolerance] = 0

    return coefficients

def compute_derivative(signal, t: float, d: int, order_accuracy: int=2) -> np.ndarray:
    '''
    Compute the derivative, d,  of a signal over a constant time/sampling interval, t, using n points prior to
    and following the frame of interest.


    Parameters
    -----------------
    signal: array_like
        an array of the signal.

    t: float
        The time interval between samples (over a vector of timestamps, it is sufficient to input t[1] -t[0]).
        Alternatively, 1/sampling frequency can also be used to denote t.

    d: int
        The derivative to take of the signal.
    
    order_accuracy: int
        The order of accuracy in which the finite difference equation will approximate to. Default = 2

    Returns
    -----------------

    signal_: ndarray
        the nth derivative of the input signal. The length of the signal is equal to that of the intput signal.
        Note: NaN values are added at beginning and/or the end of the array when insufficient data is present
        to compute the derivative (e.g. first frame of data has no prior datapoint, and thus the first n frames
        will be NaN when selecting the symmetric difference)
    '''
    # Similar to the previous function, this is just to catch any potential input errors.
    if order_accuracy % 2 == 0:
        if d % 2 == 0:
            n_pre = d + order_accuracy - 1
        else:
            n_pre = d + order_accuracy
    else:
        raise ValueError('Order of Accuracy must be a multiple of 2 for the symmetric difference')

    n = int((n_pre-1)/2)

    # Get the appropriate coefficients using the function specified above.
    coefficients = get_finite_difference_coefficients(d,order_accuracy)

    # Create an empty array that will contain the approximated derivatives as we loop through the input signal.
    derivative = np.array([])

    # Specify frames of data to be used for for loop (i.e. where to apply each coefficient)
    coefficient_num = len(np.arange(n * -1, n + 1))

    for i in range(len(signal)):
        if i < n:
            signal_prime = np.nan
        elif i >= len(signal) - n:
            signal_prime = np.nan
        else:
            # "Refresh" the signal_prime value which will be appended to the derivative vector following each loop.
            signal_prime = 0
            for c in range(coefficient_num):
                signal_prime += coefficients[c] * signal[i + c - n] / (t ** d)

        derivative = np.append(derivative, signal_prime)

    return derivative

def compute_integral_of_signal(
    original_signal: np.ndarray, sampling_frequency: float, initial_value: float=0.0
) -> np.ndarray:
    """Computes the integrated time series of a signal. For example, to compute the velocity
    waveform from an acceleration waveform

    Args:
        original_signal (np.ndarray): Original signal to integrate
        sampling_frequency (float): Sampling frequency (frames per second) of the signal
        initial_value (float, optional): Initial value when computing the instantaneous integral.
        Defaults to 0.0.

    Returns:
        np.ndarray: Integrated series
    """
    # Ensure the input is a numpy array
    if isinstance(original_signal, pd.Series):
        original_signal = original_signal.to_numpy()
    elif not isinstance(original_signal, np.ndarray):
        original_signal = np.array(original_signal)

    # Compute the time interval
    dt = 1.0 / sampling_frequency

    # Compute cumulative integral using `numpy.cumsum`
    integrated_signal = np.zeros_like(original_signal)
    integrated_signal[0] = initial_value
    integrated_signal[1:] = initial_value + np.cumsum(0.5 * (original_signal[1:] + original_signal[:-1]) * dt)

    return integrated_signal
