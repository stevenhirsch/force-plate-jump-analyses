# Signal Processing Documentation

This directory contains modules for signal processing operations, including numerical methods and filtering techniques.

## `numerical.py`

This module provides functions for numerical operations on signals, including integration, differentiation, and area calculation.

### Functions

#### integrate_area(time, signal, method='trapezoidal')
Computes the area under a curve using numerical integration.

- **Parameters:**
  - time (array): Array of timestamps
  - signal (array): Array of the signal to integrate over time
  - method (str, optional): Integration method. Either 'trapezoidal' or 'simpsons'. Defaults to 'trapezoidal'.
- **Returns:**
  - np.ndarray: Area under the curve

#### get_finite_difference_coefficients(d, order_accuracy)
Computes coefficients for finite difference equations to approximate derivatives.

- **Parameters:**
  - d (int): The order of the derivative to approximate
  - order_accuracy (int): The order of accuracy for the approximation (must be even)
- **Returns:**
  - np.ndarray: Coefficients for the finite difference equation

#### compute_derivative(signal, t, d, order_accuracy=2)
Computes the derivative of a signal using finite difference methods.

- **Parameters:**
  - signal (array_like): The input signal
  - t (float): Time interval between samples
  - d (int): The order of the derivative to compute
  - order_accuracy (int, optional): The order of accuracy. Defaults to 2.
- **Returns:**
  - np.ndarray: The computed derivative of the signal

#### compute_integral_of_signal(original_signal, sampling_frequency, initial_value=0.0)
Computes the integrated time series of a signal.

- **Parameters:**
  - original_signal (np.ndarray): Original signal to integrate
  - sampling_frequency (float): Sampling frequency in frames per second
  - initial_value (float, optional): Initial value for integration. Defaults to 0.0.
- **Returns:**
  - np.ndarray: Integrated series

## `filters.py`

This module provides filtering functions for signal processing.

### Functions

#### butterworth_filter(arr, cutoff_frequency, fps, padding, order=4)
Applies a low-pass Butterworth filter to a signal.

- **Parameters:**
  - arr (np.ndarray): Input array
  - cutoff_frequency (float): Lowpass filter cutoff frequency
  - fps (float): Frames per second of the trial
  - padding (int): Number of frames to pad the signal
  - order (int, optional): Order of the filter. Defaults to 4.
- **Returns:**
  - np.ndarray: Filtered signal