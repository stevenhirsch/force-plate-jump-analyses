"""Tests for filters.py"""
import numpy as np
import pytest
from jumpmetrics.signal_processing.filters import butterworth_filter


# Note: Unit tests were generated with Claude 3.5 Sonnet
def test_butterworth_filter_basic():
    """Basic unit test"""
    # Test with a simple sine wave
    t = np.linspace(0, 1, 1000)
    fps = 1000
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

    filtered = butterworth_filter(signal, cutoff_frequency=20, fps=fps, padding=100)

    # Check that the output has the same length as the input
    assert len(filtered) == len(signal)

    # Check that high frequencies are attenuated
    fft_original = np.fft.fft(signal)
    fft_filtered = np.fft.fft(filtered)
    assert np.max(np.abs(fft_filtered[50:])) < np.max(np.abs(fft_original[50:]))


def test_butterworth_filter_short_signal():
    """Basic unit test"""
    # Test with a signal shorter than 1 second
    short_signal = np.random.rand(500)
    fps = 1000

    filtered = butterworth_filter(short_signal, cutoff_frequency=20, fps=fps, padding=100)

    assert len(filtered) == len(short_signal)

def test_butterworth_filter_different_orders():
    """Basic unit test"""
    signal = np.random.rand(1000)
    fps = 1000

    filtered_order4 = butterworth_filter(signal, cutoff_frequency=20, fps=fps, padding=100, order=4)
    filtered_order6 = butterworth_filter(signal, cutoff_frequency=20, fps=fps, padding=100, order=6)

    assert len(filtered_order4) == len(filtered_order6) == len(signal)
    assert not np.array_equal(filtered_order4, filtered_order6)

@pytest.mark.parametrize("cutoff_frequency, fps", [
    (10, 1000),
    (50, 1000),
    (100, 2000),
])
def test_butterworth_filter_different_parameters(cutoff_frequency, fps):
    """Basic unit test"""
    signal = np.random.rand(int(fps))

    filtered = butterworth_filter(signal, cutoff_frequency=cutoff_frequency, fps=fps, padding=100)

    assert len(filtered) == len(signal)

def test_butterworth_filter_invalid_input():
    """Basic unit test"""
    with pytest.raises(ValueError):
        butterworth_filter(np.array([]), cutoff_frequency=20, fps=1000, padding=100)

def test_butterworth_filter_2d_input():
    """Basic unit test"""
    signal_2d = np.random.rand(100, 2)
    fps = 1000

    with pytest.raises(ValueError):
        butterworth_filter(signal_2d, cutoff_frequency=20, fps=fps, padding=100)
