"""Unit tests for numerical.py"""
import numpy as np
import pandas as pd
import pytest
from jumpmetrics.signal_processing.numerical import (
    integrate_area,
    get_finite_difference_coefficients,
    compute_derivative,
    compute_integral_of_signal
)


def test_integrate_area():
    """Basic unit test"""
    time = np.linspace(0, 1, 100)
    signal = np.sin(2 * np.pi * time)

    area_trap = integrate_area(time, signal, method='trapezoidal')
    area_simp = integrate_area(time, signal, method='simpsons')

    assert np.isclose(area_trap, 0, atol=1e-3)
    assert np.isclose(area_simp, 0, atol=1e-3)

    with pytest.raises(ValueError):
        integrate_area(time, signal, method='invalid')

def test_get_finite_difference_coefficients():
    """Basic unit test"""
    coeff_1st_2nd = get_finite_difference_coefficients(1, 2)
    assert np.allclose(coeff_1st_2nd, [-0.5, 0, 0.5])

    coeff_2nd_2nd = get_finite_difference_coefficients(2, 2)
    assert np.allclose(coeff_2nd_2nd, [1, -2, 1])

    with pytest.raises(ValueError):
        get_finite_difference_coefficients(1, 3)

def test_compute_derivative():
    """Basic unit test"""
    t = 0.01
    x = np.linspace(0, 1, 101)
    y = np.sin(2 * np.pi * x)

    dy_dt = compute_derivative(y, t, 1)
    d2y_dt2 = compute_derivative(y, t, 2)

    assert np.allclose(dy_dt[1:-1], 2 * np.pi * np.cos(2 * np.pi * x[1:-1]), atol=0.1)
    assert np.allclose(d2y_dt2[1:-1], -(2 * np.pi)**2 * np.sin(2 * np.pi * x[1:-1]), atol=0.1)

    assert np.isnan(dy_dt[0]) and np.isnan(dy_dt[-1])
    assert np.isnan(d2y_dt2[0]) and np.isnan(d2y_dt2[-1])

def test_compute_integral_of_signal():
    """Basic unit test"""
    t = np.linspace(0, 1, 101)
    y = np.sin(2 * np.pi * t)

    integral = compute_integral_of_signal(y, sampling_frequency=100)

    expected = -(1 / (2 * np.pi)) * np.cos(2 * np.pi * t) + (1 / (2 * np.pi))
    assert np.allclose(integral, expected, atol=1e-3)

    # Test with initial value
    integral_with_initial = compute_integral_of_signal(y, sampling_frequency=100, initial_value=1.0)
    assert np.isclose(integral_with_initial[0], 1.0)

def test_compute_integral_of_signal_with_pandas():
    """Basic unit test"""

    t = np.linspace(0, 1, 101)
    y = pd.Series(np.sin(2 * np.pi * t))

    integral = compute_integral_of_signal(y, sampling_frequency=100)

    expected = -(1 / (2 * np.pi)) * np.cos(2 * np.pi * t) + (1 / (2 * np.pi))
    assert np.allclose(integral, expected, atol=1e-3)


def test_integrate_area_edge_cases():
    """Test edge cases for integrate_area function"""

    # Test with constant signal
    time = np.linspace(0, 1, 100)
    signal = np.full(100, 5.0)

    area_trap = integrate_area(time, signal, method='trapezoidal')
    area_simp = integrate_area(time, signal, method='simpsons')

    # Area should be width * height = 1 * 5 = 5
    assert np.isclose(area_trap, 5.0, rtol=1e-3)
    assert np.isclose(area_simp, 5.0, rtol=1e-3)

    # Test with negative values
    signal_neg = np.full(100, -3.0)
    area_neg = integrate_area(time, signal_neg, method='trapezoidal')
    assert np.isclose(area_neg, -3.0, rtol=1e-3)

    # Test with single point (should return 0)
    time_single = np.array([0])
    signal_single = np.array([5])
    area_single = integrate_area(time_single, signal_single, method='trapezoidal')
    assert area_single == 0

    # Test with two points
    time_two = np.array([0, 1])
    signal_two = np.array([0, 2])
    area_two = integrate_area(time_two, signal_two, method='trapezoidal')
    assert np.isclose(area_two, 1.0)  # Triangle area = 0.5 * base * height


def test_get_finite_difference_coefficients_edge_cases():
    """Test edge cases for finite difference coefficients"""

    # ACTUAL BEHAVIOR: Function requires even order_accuracy for symmetric difference
    # Test that odd order accuracy raises appropriate error
    with pytest.raises(ValueError, match="Order of Accuracy must be a multiple of 2"):
        get_finite_difference_coefficients(1, 1)  # Odd order not allowed

    with pytest.raises(ValueError, match="Order of Accuracy must be a multiple of 2"):
        get_finite_difference_coefficients(1, 3)  # Odd order not allowed

    # Test valid even order accuracy
    coeff_1st_2nd = get_finite_difference_coefficients(1, 2)
    assert len(coeff_1st_2nd) == 3  # Should have 3 coefficients

    # Test higher derivatives with appropriate even accuracy
    coeff_3rd_4th = get_finite_difference_coefficients(3, 4)
    assert len(coeff_3rd_4th) == 7  # Should have 7 coefficients for 3rd derivative, 4th order

    # Test second derivative with even order
    coeff_2nd_2nd = get_finite_difference_coefficients(2, 2)
    assert len(coeff_2nd_2nd) == 3  # Should have 3 coefficients


def test_compute_derivative_edge_cases():
    """Test edge cases for compute_derivative function"""

    # Test with constant signal (derivative should be zero)
    y_constant = np.full(10, 5.0)
    t = 0.1
    dy_dt = compute_derivative(y_constant, t, 1)

    # Interior points should be approximately zero
    assert np.allclose(dy_dt[1:-1], 0, atol=1e-10)
    # Boundary points should be NaN
    assert np.isnan(dy_dt[0]) and np.isnan(dy_dt[-1])

    # Test with linear signal (derivative should be constant)
    y_linear = np.linspace(0, 10, 11)  # Slope = 1
    dy_dt_linear = compute_derivative(y_linear, 1.0, 1)

    # Interior points should be approximately 1
    assert np.allclose(dy_dt_linear[1:-1], 1.0, atol=1e-10)

    # Test second derivative of quadratic (should be constant)
    x = np.linspace(0, 1, 11)
    y_quad = x**2  # d²y/dx² = 2
    d2y_dx2 = compute_derivative(y_quad, 0.1, 2)

    # Interior points should be approximately 2
    assert np.allclose(d2y_dx2[1:-1], 2.0, atol=1e-1)

    # Test with very small time step
    dy_dt_small = compute_derivative(y_linear, 1e-6, 1)
    assert np.allclose(dy_dt_small[1:-1], 1e6, atol=1e-4)  # Slope/dt

    # Test with negative values
    y_neg = -np.linspace(0, 10, 11)
    dy_dt_neg = compute_derivative(y_neg, 1.0, 1)
    assert np.allclose(dy_dt_neg[1:-1], -1.0, atol=1e-10)


def test_compute_integral_of_signal_edge_cases():
    """Test edge cases for compute_integral_of_signal function"""

    # Test with constant signal
    y_constant = np.full(100, 3.0)
    integral_constant = compute_integral_of_signal(y_constant, sampling_frequency=100)

    # Integral of constant should be linear
    expected_linear = 3.0 * np.arange(100) / 100
    assert np.allclose(integral_constant, expected_linear, atol=1e-10)

    # Test with zero signal
    y_zero = np.zeros(100)
    integral_zero = compute_integral_of_signal(y_zero, sampling_frequency=100)
    assert np.allclose(integral_zero, 0, atol=1e-10)

    # Test with single sample
    y_single = np.array([5.0])
    integral_single = compute_integral_of_signal(y_single, sampling_frequency=100)
    assert len(integral_single) == 1
    assert integral_single[0] == 0  # First sample should be initial value

    # Test with very low sampling frequency
    y = np.array([1, 2, 3, 4])
    integral_low_freq = compute_integral_of_signal(y, sampling_frequency=1)
    # Should produce larger increments
    assert len(integral_low_freq) == len(y)

    # Test with very high sampling frequency
    y = np.array([1, 2, 3, 4])
    integral_high_freq = compute_integral_of_signal(y, sampling_frequency=1e6)
    # Should produce very small increments
    assert np.all(np.diff(integral_high_freq) >= 0)  # Should be monotonic for positive input

    # Test with custom initial value
    y = np.array([1, 1, 1, 1])
    initial_value = 10.0
    integral_custom = compute_integral_of_signal(y, sampling_frequency=100, initial_value=initial_value)
    assert integral_custom[0] == initial_value

    # Test with alternating signal (should oscillate around initial value)
    y_alternating = np.array([1, -1, 1, -1, 1, -1])
    integral_alt = compute_integral_of_signal(y_alternating, sampling_frequency=100)
    # Should return close to initial value at end for symmetric alternating signal
    assert abs(integral_alt[-1]) < 0.1


def test_numerical_functions_with_extreme_values():
    """Test numerical functions with extreme values"""

    # Test with very large values
    large_signal = np.full(100, 1e10)
    large_time = np.linspace(0, 1, 100)
    large_area = integrate_area(large_time, large_signal, method='trapezoidal')
    assert np.isfinite(large_area)
    assert large_area > 0

    # Test with very small values
    small_signal = np.full(100, 1e-10)
    small_area = integrate_area(large_time, small_signal, method='trapezoidal')
    assert np.isfinite(small_area)
    assert small_area > 0

    # Test derivative with extreme values
    extreme_y = np.array([1e10, 1e10, 1e10])
    extreme_deriv = compute_derivative(extreme_y, 1.0, 1)
    assert np.isfinite(extreme_deriv[1])  # Middle point should be finite

    # Test integral with extreme values
    extreme_integral = compute_integral_of_signal(large_signal, sampling_frequency=100)
    assert np.all(np.isfinite(extreme_integral))


def test_numerical_functions_with_nan_inf():
    """Test handling of NaN and infinite values"""

    # Test integrate_area with NaN
    time = np.linspace(0, 1, 100)
    signal_with_nan = np.full(100, 1.0)
    signal_with_nan[50] = np.nan

    area_nan = integrate_area(time, signal_with_nan, method='trapezoidal')
    assert np.isnan(area_nan)

    # Test integrate_area with infinity
    signal_with_inf = np.full(100, 1.0)
    signal_with_inf[50] = np.inf

    area_inf = integrate_area(time, signal_with_inf, method='trapezoidal')
    assert np.isinf(area_inf)

    # Test derivative with NaN
    y_with_nan = np.array([1, 2, np.nan, 4, 5])
    deriv_nan = compute_derivative(y_with_nan, 1.0, 1)
    assert np.isnan(deriv_nan[2])  # Should propagate NaN

    # Test integral with NaN
    integral_nan = compute_integral_of_signal(y_with_nan, sampling_frequency=100)
    # Once NaN appears, it should propagate
    nan_index = np.where(np.isnan(y_with_nan))[0][0]
    assert np.all(np.isnan(integral_nan[nan_index+1:]))
