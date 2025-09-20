"""Comprehensive unit tests for metrics.metrics module"""
import pytest
import numpy as np
from unittest.mock import patch
from jumpmetrics.metrics.metrics import (
    get_bodyweight,
    compute_rfd,
    compute_jump_height_from_takeoff_velocity,
    compute_jump_height_from_velocity_series,
    compute_jump_height_from_net_vertical_impulse,
    compute_average_force_between_events,
    compute_jump_height_from_total_flight_time,
    compute_jump_height_from_flight_time_events
)


class TestGetBodyweight:
    """Test get_bodyweight function"""

    def test_normal_bodyweight_calculation(self):
        """Test normal bodyweight calculation"""
        force_series = np.array([1000, 1000, 1000, 1000, 1000, 800, 1200])
        sampling_frequency = 1000
        sec = 0.005  # Use first 5 samples

        result = get_bodyweight(force_series, sampling_frequency, sec)

        assert result == 1000.0

    def test_noisy_bodyweight_calculation(self):
        """Test bodyweight calculation with noisy data"""
        # Create noisy force data
        np.random.seed(42)
        base_force = 980
        noise = np.random.normal(0, 10, 500)
        force_series = base_force + noise

        result = get_bodyweight(force_series, sampling_frequency=1000, sec=0.5)

        # Should be close to base force despite noise
        assert abs(result - base_force) < 20

    def test_different_weighing_times(self):
        """Test with different weighing time durations"""
        force_series = np.array([1000] * 1000 + [500] * 1000)  # 1s stable, then drop

        # Short weighing time
        result_short = get_bodyweight(force_series, sampling_frequency=1000, sec=0.1)
        assert result_short == 1000.0

        # Long weighing time (should still be stable part)
        result_long = get_bodyweight(force_series, sampling_frequency=1000, sec=0.5)
        assert result_long == 1000.0

    def test_very_short_weighing_time(self):
        """Test with very short weighing time"""
        force_series = np.array([950, 1000, 1050, 980, 1020])

        result = get_bodyweight(force_series, sampling_frequency=1000, sec=0.001)

        # Should use only first sample
        assert result == 950.0

    def test_fractional_sample_count(self):
        """Test when sec * sampling_frequency gives fractional samples"""
        force_series = np.array([1000, 1010, 1020, 1030, 1040])

        # 0.0035s * 1000Hz = 3.5 samples, should truncate to 3
        result = get_bodyweight(force_series, sampling_frequency=1000, sec=0.0035)

        expected = np.mean([1000, 1010, 1020])  # First 3 samples
        assert result == expected

    def test_empty_force_series(self):
        """Test with empty force series"""
        force_series = np.array([])

        # Empty array will result in warning and NaN
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            result = get_bodyweight(force_series, sampling_frequency=1000, sec=0.1)
            assert np.isnan(result)

    def test_single_sample(self):
        """Test with single sample"""
        force_series = np.array([985])

        result = get_bodyweight(force_series, sampling_frequency=1000, sec=0.001)

        assert result == 985.0


class TestComputeRfd:
    """Test compute_rfd function"""

    def test_average_rfd_calculation(self):
        """Test average RFD calculation"""
        force_trace = np.array([1000, 1100, 1200, 1300, 1400])
        window_start = 1
        window_end = 4
        sampling_frequency = 1000

        result = compute_rfd(
            force_trace, window_start, window_end,
            sampling_frequency, method='average'
        )

        # (1400 - 1100) / (3/1000) = 300 / 0.003 = 100000 N/s
        expected = 100000.0
        assert result == expected

    def test_instantaneous_rfd_calculation(self):
        """Test instantaneous RFD calculation"""
        force_trace = np.array([1000, 1100, 1200, 1300, 1400])
        window_start = 1
        window_end = 4
        sampling_frequency = 1000

        with patch('jumpmetrics.metrics.metrics.compute_derivative') as mock_derivative:
            mock_derivative.return_value = np.array([np.nan, 100000, 100000, 100000, np.nan])

            result = compute_rfd(
                force_trace, window_start, window_end,
                sampling_frequency, method='instantaneous'
            )

            assert result == 100000.0  # Mean of non-NaN values

    def test_peak_rfd_calculation(self):
        """Test peak RFD calculation"""
        force_trace = np.array([1000, 1100, 1200, 1300, 1400])
        window_start = 1
        window_end = 4
        sampling_frequency = 1000

        with patch('jumpmetrics.metrics.metrics.compute_derivative') as mock_derivative:
            mock_derivative.return_value = np.array([np.nan, 80000, 120000, 100000, np.nan])

            result = compute_rfd(
                force_trace, window_start, window_end,
                sampling_frequency, method='peak'
            )

            assert result == 120000.0  # Maximum value

    def test_rfd_with_not_found_window_start(self):
        """Test RFD calculation when window start is NOT_FOUND (-100)"""
        force_trace = np.array([1000, 1100, 1200])
        window_start = -100  # NOT_FOUND
        window_end = 2
        sampling_frequency = 1000

        with patch('logging.warning') as mock_warning:
            result = compute_rfd(
                force_trace, window_start, window_end,
                sampling_frequency, method='average'
            )
            mock_warning.assert_called_once()

        assert np.isnan(result)

    def test_rfd_with_not_found_window_end(self):
        """Test RFD calculation when window end is NOT_FOUND (-100)"""
        force_trace = np.array([1000, 1100, 1200])
        window_start = 0
        window_end = -100  # NOT_FOUND
        sampling_frequency = 1000

        with patch('logging.warning') as mock_warning:
            result = compute_rfd(
                force_trace, window_start, window_end,
                sampling_frequency, method='average'
            )
            mock_warning.assert_called_once()

        assert np.isnan(result)

    def test_rfd_with_invalid_window_order(self):
        """Test RFD calculation when window_end <= window_start"""
        force_trace = np.array([1000, 1100, 1200, 1300])
        window_start = 2
        window_end = 1  # Before start
        sampling_frequency = 1000

        with patch('logging.warning') as mock_warning:
            result = compute_rfd(
                force_trace, window_start, window_end,
                sampling_frequency, method='average'
            )
            mock_warning.assert_called_once()

        assert np.isnan(result)

    def test_rfd_with_equal_windows(self):
        """Test RFD calculation when window_end == window_start"""
        force_trace = np.array([1000, 1100, 1200, 1300])
        window_start = 2
        window_end = 2  # Equal to start
        sampling_frequency = 1000

        with patch('logging.warning') as mock_warning:
            result = compute_rfd(
                force_trace, window_start, window_end,
                sampling_frequency, method='average'
            )
            mock_warning.assert_called_once()

        assert np.isnan(result)

    def test_rfd_with_invalid_method(self):
        """Test RFD calculation with invalid method"""
        force_trace = np.array([1000, 1100, 1200, 1300])
        window_start = 0
        window_end = 2
        sampling_frequency = 1000

        with patch('logging.error') as mock_error:
            result = compute_rfd(
                force_trace, window_start, window_end,
                sampling_frequency, method='invalid_method'
            )
            mock_error.assert_called_once()

        assert result is None


class TestJumpHeightCalculations:
    """Test various jump height calculation functions"""

    def test_jump_height_from_takeoff_velocity(self):
        """Test jump height calculation from takeoff velocity"""
        takeoff_velocity = 3.13  # m/s (should give ~0.5m height)

        result = compute_jump_height_from_takeoff_velocity(takeoff_velocity)

        expected = (3.13 ** 2) / (2 * 9.81)
        assert abs(result - expected) < 1e-10

    def test_jump_height_from_velocity_series(self):
        """Test jump height calculation from velocity series"""
        velocity_series = np.array([0, 1.0, 2.0, 2.5, 3.0])  # Last value = takeoff velocity

        result = compute_jump_height_from_velocity_series(velocity_series)

        expected = (3.0 ** 2) / (2 * 9.81)
        assert abs(result - expected) < 1e-10

    def test_jump_height_from_net_vertical_impulse(self):
        """Test jump height calculation from net vertical impulse"""
        net_vertical_impulse = 156.5  # N⋅s
        body_mass_kg = 75.0  # kg

        result = compute_jump_height_from_net_vertical_impulse(
            net_vertical_impulse, body_mass_kg
        )

        # impulse/mass = velocity, then use kinematic equation
        takeoff_velocity = net_vertical_impulse / body_mass_kg
        expected = (takeoff_velocity ** 2) / (2 * 9.81)
        assert abs(result - expected) < 1e-10

    def test_jump_height_from_total_flight_time(self):
        """Test jump height calculation from total flight time"""
        flight_time = 0.6  # seconds

        result = compute_jump_height_from_total_flight_time(flight_time)

        # h = 0.5 * g * (t/2)^2
        expected = 0.5 * 9.81 * (0.3 ** 2)
        assert abs(result - expected) < 1e-10

    def test_jump_height_from_flight_time_events(self):
        """Test jump height calculation from takeoff and landing frame events"""
        takeoff_frame = 1000
        landing_frame = 1600  # 600 frames later
        sampling_frequency = 1000  # 1000 Hz

        result = compute_jump_height_from_flight_time_events(
            takeoff_frame, landing_frame, sampling_frequency
        )

        # 600 frames / 1000 Hz = 0.6s flight time
        flight_time = 0.6
        expected = 0.5 * 9.81 * (0.3 ** 2)
        assert abs(result - expected) < 1e-10

    def test_zero_takeoff_velocity(self):
        """Test jump height with zero takeoff velocity"""
        result = compute_jump_height_from_takeoff_velocity(0.0)
        assert result == 0.0

    def test_negative_takeoff_velocity(self):
        """Test jump height with negative takeoff velocity"""
        # Technically possible in some calculations due to numerical errors
        result = compute_jump_height_from_takeoff_velocity(-1.0)
        expected = ((-1.0) ** 2) / (2 * 9.81)
        assert abs(result - expected) < 1e-10

    def test_very_high_takeoff_velocity(self):
        """Test jump height with unrealistically high takeoff velocity"""
        takeoff_velocity = 10.0  # Very high
        result = compute_jump_height_from_takeoff_velocity(takeoff_velocity)
        expected = (10.0 ** 2) / (2 * 9.81)
        assert abs(result - expected) < 1e-10


class TestComputeAverageForceBetweenEvents:
    """Test compute_average_force_between_events function"""

    def test_normal_average_force_calculation(self):
        """Test normal average force calculation"""
        force_trace = np.array([1000, 1100, 1200, 1300, 1400, 1200, 1000])
        window_start = 1
        window_end = 5

        result = compute_average_force_between_events(
            force_trace, window_start, window_end
        )

        expected = np.mean([1100, 1200, 1300, 1400])  # Indices 1:5
        assert result == expected

    def test_average_force_with_not_found_start(self):
        """Test average force when window start is NOT_FOUND"""
        force_trace = np.array([1000, 1100, 1200])
        window_start = -100  # NOT_FOUND
        window_end = 2

        with patch('logging.warning') as mock_warning:
            result = compute_average_force_between_events(
                force_trace, window_start, window_end
            )
            mock_warning.assert_called_once()

        assert np.isnan(result)

    def test_average_force_with_not_found_end(self):
        """Test average force when window end is NOT_FOUND"""
        force_trace = np.array([1000, 1100, 1200])
        window_start = 0
        window_end = -100  # NOT_FOUND

        with patch('logging.warning') as mock_warning:
            result = compute_average_force_between_events(
                force_trace, window_start, window_end
            )
            mock_warning.assert_called_once()

        assert np.isnan(result)

    def test_average_force_with_invalid_window_order(self):
        """Test average force when window_end <= window_start"""
        force_trace = np.array([1000, 1100, 1200, 1300])
        window_start = 2
        window_end = 1  # Before start

        with patch('logging.warning') as mock_warning:
            result = compute_average_force_between_events(
                force_trace, window_start, window_end
            )
            mock_warning.assert_called_once()

        assert np.isnan(result)

    def test_average_force_with_equal_windows(self):
        """Test average force when window_end == window_start (positive values)"""
        force_trace = np.array([1000, 1100, 1200, 1300])
        window_start = 1
        window_end = 1  # Equal, but both positive

        with patch('logging.warning') as mock_warning:
            result = compute_average_force_between_events(
                force_trace, window_start, window_end
            )
            mock_warning.assert_called_once()

        assert np.isnan(result)

    def test_average_force_single_sample_window(self):
        """Test average force with effectively single sample window"""
        force_trace = np.array([1000, 1100, 1200, 1300])
        window_start = 1
        window_end = 2  # Only one sample in window

        result = compute_average_force_between_events(
            force_trace, window_start, window_end
        )

        # Should return the single value (index 1)
        assert result == 1100

    def test_average_force_with_negative_values(self):
        """Test average force calculation with negative force values"""
        force_trace = np.array([-100, -50, 0, 50, 100])
        window_start = 1
        window_end = 4

        result = compute_average_force_between_events(
            force_trace, window_start, window_end
        )

        expected = np.mean([-50, 0, 50])  # Indices 1:4
        assert result == expected

    def test_average_force_with_floating_point_values(self):
        """Test average force with floating point force values"""
        force_trace = np.array([1000.5, 1100.7, 1200.3, 1300.9])
        window_start = 0
        window_end = 3

        result = compute_average_force_between_events(
            force_trace, window_start, window_end
        )

        expected = np.mean([1000.5, 1100.7, 1200.3])
        assert abs(result - expected) < 1e-10

    def test_average_force_with_zero_values(self):
        """Test average force calculation with zero force values"""
        force_trace = np.array([0, 0, 0, 0])
        window_start = 0
        window_end = 3

        result = compute_average_force_between_events(
            force_trace, window_start, window_end
        )

        assert result == 0.0

    def test_average_force_with_large_window(self):
        """Test average force with large window spanning most of the trace"""
        force_trace = np.arange(1000, 2000, 10)  # 100 values from 1000 to 1990
        window_start = 10
        window_end = 90

        result = compute_average_force_between_events(
            force_trace, window_start, window_end
        )

        expected = np.mean(force_trace[10:90])
        assert result == expected