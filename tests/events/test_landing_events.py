"""Comprehensive unit tests for events.landing_events module"""
import pytest
import numpy as np
from jumpmetrics.events.landing_events import get_end_of_landing_phase


class TestGetEndOfLandingPhase:
    """Test get_end_of_landing_phase function"""

    def test_normal_landing_phase_end(self):
        """Test normal case where landing phase end is clearly detected"""
        # Create velocity series showing landing (negative) transitioning to positive
        velocity_series = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0])

        result = get_end_of_landing_phase(velocity_series)

        # Should detect end at index 4 (first zero or positive velocity)
        assert result == 4

    def test_immediate_positive_velocity(self):
        """Test when velocity is positive from the start"""
        velocity_series = np.array([0.5, 1.0, 1.5, 2.0])

        result = get_end_of_landing_phase(velocity_series)

        # Should return first index (0)
        assert result == 0

    def test_zero_velocity_at_start(self):
        """Test when velocity starts at zero"""
        velocity_series = np.array([0.0, 0.5, 1.0, 1.5])

        result = get_end_of_landing_phase(velocity_series)

        # Should return first index (0) since zero counts as >= 0
        assert result == 0

    def test_all_negative_velocities(self):
        """Test when all velocities are negative (no end of landing detected)"""
        velocity_series = np.array([-2.0, -1.5, -1.0, -0.5, -0.1])

        result = get_end_of_landing_phase(velocity_series)

        # Should return -1 (not found)
        assert result == -1

    def test_multiple_zero_crossings(self):
        """Test when velocity crosses zero multiple times"""
        velocity_series = np.array([-1.0, -0.5, 0.0, -0.2, 0.5, 1.0])

        result = get_end_of_landing_phase(velocity_series)

        # Should return first occurrence of zero or positive velocity
        assert result == 2

    def test_exact_zero_velocity(self):
        """Test when velocity is exactly zero"""
        velocity_series = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

        result = get_end_of_landing_phase(velocity_series)

        # Should detect zero as end of landing phase
        assert result == 2

    def test_very_small_positive_velocity(self):
        """Test with very small positive velocity"""
        velocity_series = np.array([-1.0, -0.5, 1e-10, 0.5, 1.0])

        result = get_end_of_landing_phase(velocity_series)

        # Should detect very small positive velocity
        assert result == 2

    def test_empty_velocity_series(self):
        """Test with empty velocity series"""
        velocity_series = np.array([])

        result = get_end_of_landing_phase(velocity_series)

        # Should return -1 for empty array
        assert result == -1

    def test_single_value_negative(self):
        """Test with single negative velocity value"""
        velocity_series = np.array([-1.0])

        result = get_end_of_landing_phase(velocity_series)

        # Should return -1 (no positive values)
        assert result == -1

    def test_single_value_positive(self):
        """Test with single positive velocity value"""
        velocity_series = np.array([1.0])

        result = get_end_of_landing_phase(velocity_series)

        # Should return 0 (first and only positive value)
        assert result == 0

    def test_single_value_zero(self):
        """Test with single zero velocity value"""
        velocity_series = np.array([0.0])

        result = get_end_of_landing_phase(velocity_series)

        # Should return 0 (zero counts as >= 0)
        assert result == 0

    def test_realistic_landing_velocity_profile(self):
        """Test with realistic landing velocity profile"""
        # Simulate realistic landing: high negative velocity decreasing to positive
        velocity_series = np.array([
            -3.5, -3.2, -2.8, -2.3, -1.8, -1.2, -0.7, -0.3, -0.1, 0.0, 0.2, 0.5, 0.8
        ])

        result = get_end_of_landing_phase(velocity_series)

        # Should detect end at zero velocity (index 9)
        assert result == 9

    def test_noisy_velocity_data(self):
        """Test with noisy velocity data around zero crossing"""
        # Simulate noise around the zero crossing
        velocity_series = np.array([-0.5, -0.2, -0.05, 0.01, -0.01, 0.1, 0.3])

        result = get_end_of_landing_phase(velocity_series)

        # Should detect first positive value (even if small)
        assert result == 3

    def test_velocity_with_floating_point_precision(self):
        """Test with floating point precision issues near zero"""
        # Create velocities that might have floating point precision issues
        velocity_series = np.array([-0.1, -1e-15, 1e-16, 0.1])

        result = get_end_of_landing_phase(velocity_series)

        # Should handle floating point precision correctly
        # Both very small negative and positive should be detected properly
        assert result == 2  # First value >= 0 (1e-16)

    def test_all_zero_velocities(self):
        """Test when all velocities are zero"""
        velocity_series = np.array([0.0, 0.0, 0.0, 0.0])

        result = get_end_of_landing_phase(velocity_series)

        # Should return first index (0)
        assert result == 0

    def test_mixed_positive_negative_pattern(self):
        """Test complex pattern of positive and negative velocities"""
        velocity_series = np.array([-1.0, 0.5, -0.3, 0.2, 1.0])

        result = get_end_of_landing_phase(velocity_series)

        # Should return first positive velocity
        assert result == 1

    def test_pandas_series_input(self):
        """Test that function works with pandas Series input"""
        import pandas as pd
        velocity_series = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.0])

        result = get_end_of_landing_phase(velocity_series)

        # Should work the same as numpy array
        assert result == 2

    def test_velocity_series_with_nan_values(self):
        """Test handling of NaN values in velocity series"""
        velocity_series = np.array([-1.0, np.nan, 0.0, 0.5])

        result = get_end_of_landing_phase(velocity_series)

        # Should skip NaN and find first valid positive value
        assert result == 2

    def test_velocity_series_with_inf_values(self):
        """Test handling of infinite values"""
        velocity_series = np.array([-1.0, -np.inf, np.inf, 0.5])

        result = get_end_of_landing_phase(velocity_series)

        # Should detect positive infinity as >= 0
        assert result == 2