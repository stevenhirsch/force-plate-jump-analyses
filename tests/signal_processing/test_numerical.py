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
