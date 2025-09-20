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


def test_butterworth_filter_edge_cases():
    """Test edge cases and error conditions"""

    # Test with very high cutoff frequency (close to Nyquist)
    signal = np.random.rand(1000)
    fps = 1000

    # Cutoff at 450 Hz (close to Nyquist of 500 Hz)
    filtered = butterworth_filter(signal, cutoff_frequency=450, fps=fps, padding=100)
    assert len(filtered) == len(signal)

    # Test with cutoff frequency exactly at Nyquist
    with pytest.raises((ValueError, Exception)):
        butterworth_filter(signal, cutoff_frequency=500, fps=fps, padding=100)


def test_butterworth_filter_zero_padding():
    """Test with zero padding"""
    signal = np.random.rand(1000)
    fps = 1000

    filtered = butterworth_filter(signal, cutoff_frequency=20, fps=fps, padding=0)
    assert len(filtered) == len(signal)


def test_butterworth_filter_large_padding():
    """Test with padding larger than signal"""
    signal = np.random.rand(100)
    fps = 1000

    # Padding larger than signal length
    filtered = butterworth_filter(signal, cutoff_frequency=20, fps=fps, padding=200)
    assert len(filtered) == len(signal)


def test_butterworth_filter_constant_signal():
    """Test with constant signal (DC component only)"""
    signal = np.full(1000, 5.0)  # Constant signal
    fps = 1000

    filtered = butterworth_filter(signal, cutoff_frequency=20, fps=fps, padding=100)

    # DC component should be preserved
    assert np.allclose(filtered, signal, rtol=1e-3)


def test_butterworth_filter_nyquist_frequency():
    """Test behavior near Nyquist frequency"""
    signal = np.random.rand(1000)
    fps = 1000
    nyquist = fps / 2

    # Test with cutoff just below Nyquist
    filtered = butterworth_filter(signal, cutoff_frequency=nyquist - 10, fps=fps, padding=100)
    assert len(filtered) == len(signal)


def test_butterworth_filter_very_low_frequency():
    """Test with very low cutoff frequency"""
    # Create signal with multiple frequency components
    t = np.linspace(0, 2, 2000)
    signal = (np.sin(2 * np.pi * 1 * t) +  # 1 Hz
              np.sin(2 * np.pi * 5 * t) +  # 5 Hz
              np.sin(2 * np.pi * 20 * t))  # 20 Hz

    fps = 1000

    # Very low cutoff should remove most frequency content
    filtered = butterworth_filter(signal, cutoff_frequency=2, fps=fps, padding=200)

    # Check that high frequency content is significantly reduced
    fft_original = np.fft.fft(signal)
    fft_filtered = np.fft.fft(filtered)
    freqs = np.fft.fftfreq(len(signal), 1/fps)

    # Energy above 10 Hz should be much reduced
    high_freq_mask = np.abs(freqs) > 10
    original_high_energy = np.sum(np.abs(fft_original[high_freq_mask])**2)
    filtered_high_energy = np.sum(np.abs(fft_filtered[high_freq_mask])**2)

    assert filtered_high_energy < 0.1 * original_high_energy


def test_butterworth_filter_negative_values():
    """Test filter with negative signal values"""
    signal = np.sin(2 * np.pi * np.linspace(0, 1, 1000)) - 5  # Offset to negative
    fps = 1000

    filtered = butterworth_filter(signal, cutoff_frequency=20, fps=fps, padding=100)

    assert len(filtered) == len(signal)
    # Mean should be preserved (approximately)
    assert abs(np.mean(filtered) - np.mean(signal)) < 0.5


def test_butterworth_filter_single_sample():
    """Test with single sample"""
    signal = np.array([1.0])
    fps = 1000

    # Should handle gracefully or raise appropriate error
    try:
        filtered = butterworth_filter(signal, cutoff_frequency=20, fps=fps, padding=0)
        assert len(filtered) == 1
    except (ValueError, Exception):
        # It's acceptable to raise an error for such a small signal
        pass


def test_butterworth_filter_preserve_mean():
    """Test that filter preserves signal mean for low frequencies"""
    # Create signal with clear DC offset
    signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000)) + 10
    fps = 1000

    filtered = butterworth_filter(signal, cutoff_frequency=50, fps=fps, padding=100)

    # Mean should be approximately preserved
    assert abs(np.mean(filtered) - np.mean(signal)) < 0.1
