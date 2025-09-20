"""Comprehensive unit tests for core.io module"""
import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, mock_open
from jumpmetrics.core.io import (
    load_cropped_force_data,
    load_raw_force_data,
    load_raw_force_data_with_no_column_headers,
    sum_dual_force_components,
    find_first_frame_where_force_exceeds_threshold,
    find_frame_when_off_plate,
    get_n_seconds_before_takeoff,
    find_landing_frame
)


class TestLoadCroppedForceData:
    """Test load_cropped_force_data function"""

    def test_load_cropped_force_data_with_freq(self):
        """Test loading force data with specific frequency"""
        test_data = "column1\tFZsum_26.0Hz\tcolumn3\n1\t100.5\t3\n2\t200.7\t4\n"

        with patch("builtins.open", mock_open(read_data=test_data)):
            with patch("pandas.read_csv") as mock_read_csv:
                mock_df = pd.DataFrame({'FZsum_26.0Hz': [100.5, 200.7]})
                mock_read_csv.return_value = mock_df

                result = load_cropped_force_data("test.txt", freq=26.0)

                assert isinstance(result, pd.Series)
                assert len(result) == 2
                assert result.iloc[0] == 100.5
                assert result.iloc[1] == 200.7

    def test_load_cropped_force_data_without_freq(self):
        """Test loading force data without frequency (raw data)"""
        test_data = "column1\tFZsum\tcolumn3\n1\t100.5\t3\n2\t200.7\t4\n"

        with patch("builtins.open", mock_open(read_data=test_data)):
            with patch("pandas.read_csv") as mock_read_csv:
                mock_df = pd.DataFrame({'FZsum': [100.5, 200.7]})
                mock_read_csv.return_value = mock_df

                result = load_cropped_force_data("test.txt", freq=None)

                assert isinstance(result, pd.Series)
                assert len(result) == 2

    def test_load_cropped_force_data_file_not_found(self):
        """Test handling of file not found error"""
        with patch("pandas.read_csv", side_effect=FileNotFoundError):
            result = load_cropped_force_data("nonexistent.txt")
            assert result is None

    def test_load_cropped_force_data_parser_error(self):
        """Test handling of parser error"""
        with patch("pandas.read_csv", side_effect=pd.errors.ParserError("Invalid format")):
            result = load_cropped_force_data("invalid.txt")
            assert result is None

    def test_load_cropped_force_data_missing_column(self):
        """Test handling of missing column"""
        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame({'other_column': [1, 2, 3]})
            mock_read_csv.return_value = mock_df

            result = load_cropped_force_data("test.txt", freq=26.0)
            assert result is None

    def test_load_cropped_force_data_key_error(self):
        """Test handling of key error when accessing column"""
        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame({'other_column': [1, 2, 3]})  # Missing FZsum column
            mock_read_csv.return_value = mock_df

            result = load_cropped_force_data("test.txt")
            assert result is None


class TestLoadRawForceData:
    """Test load_raw_force_data function"""

    def test_load_raw_force_data_success(self):
        """Test successful loading of raw force data"""
        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame({
                'FZ1': [100, 200, 300],
                'FZ2': [150, 250, 350],
                'Unnamed: 0': [0, 1, 2]
            })
            mock_read_csv.return_value = mock_df

            result = load_raw_force_data("test.txt")

            assert isinstance(result, pd.DataFrame)
            assert 'Unnamed: 0' not in result.columns
            assert 'FZ1' in result.columns
            assert 'FZ2' in result.columns

    def test_load_raw_force_data_without_unnamed_column(self):
        """Test loading data without 'Unnamed: 0' column"""
        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame({
                'FZ1': [100, 200, 300],
                'FZ2': [150, 250, 350]
            })
            mock_read_csv.return_value = mock_df

            result = load_raw_force_data("test.txt")

            assert isinstance(result, pd.DataFrame)
            assert len(result.columns) == 2

    def test_load_raw_force_data_file_not_found(self):
        """Test handling of file not found error"""
        with patch("pandas.read_csv", side_effect=FileNotFoundError):
            result = load_raw_force_data("nonexistent.txt")
            assert result is None

    def test_load_raw_force_data_parser_error(self):
        """Test handling of parser error"""
        with patch("pandas.read_csv", side_effect=pd.errors.ParserError("Invalid format")):
            result = load_raw_force_data("invalid.txt")
            assert result is None


class TestLoadRawForceDataWithNoColumnHeaders:
    """Test load_raw_force_data_with_no_column_headers function"""

    def test_load_success(self):
        """Test successful loading with proper column assignment"""
        test_data = "\n100, 200, 300, 10, 20, 30, 150, 250, 350, 15, 25, 35\n110, 210, 310, 11, 21, 31, 160, 260, 360, 16, 26, 36"

        with patch("builtins.open", mock_open(read_data=test_data)):
            with patch("pandas.read_csv") as mock_read_csv:
                mock_df = pd.DataFrame([
                    [100, 200, 300, 10, 20, 30, 150, 250, 350, 15, 25, 35],
                    [110, 210, 310, 11, 21, 31, 160, 260, 360, 16, 26, 36]
                ])
                mock_read_csv.return_value = mock_df

                result = load_raw_force_data_with_no_column_headers("test.txt")

                expected_cols = ['FX1', 'FY1', 'FZ1', 'MX1', 'MY1', 'MZ1', 'FX2', 'FY2', 'FZ2', 'MX2', 'MY2', 'MZ2']
                assert isinstance(result, pd.DataFrame)
                assert list(result.columns) == expected_cols

    def test_load_with_whitespace_handling(self):
        """Test proper handling of whitespace in data"""
        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame([
                [' 100 ', ' 200 ', 300, 10, 20, 30, 150, 250, 350, 15, 25, 35]
            ])
            mock_read_csv.return_value = mock_df

            with patch.object(mock_df, 'applymap') as mock_applymap:
                mock_applymap.return_value = mock_df
                result = load_raw_force_data_with_no_column_headers("test.txt")
                mock_applymap.assert_called_once()

    def test_load_file_not_found(self):
        """Test handling of file not found error"""
        with patch("pandas.read_csv", side_effect=FileNotFoundError):
            result = load_raw_force_data_with_no_column_headers("nonexistent.txt")
            assert result is None

    def test_load_parser_error(self):
        """Test handling of parser error"""
        with patch("pandas.read_csv", side_effect=pd.errors.ParserError("Invalid format")):
            result = load_raw_force_data_with_no_column_headers("invalid.txt")
            assert result is None


class TestSumDualForceComponents:
    """Test sum_dual_force_components function"""

    def test_sum_default_components(self):
        """Test summing with default FZ1 and FZ2 components"""
        df = pd.DataFrame({
            'FZ1': [100, 200, 300],
            'FZ2': [50, 75, 100]
        })

        result = sum_dual_force_components(df)

        expected = pd.Series([150, 275, 400])
        pd.testing.assert_series_equal(result, expected)

    def test_sum_custom_components(self):
        """Test summing with custom component names"""
        df = pd.DataFrame({
            'FX1': [10, 20, 30],
            'FX2': [5, 7.5, 10]
        })

        result = sum_dual_force_components(df, component_1='FX1', component_2='FX2')

        expected = pd.Series([15, 27.5, 40])
        pd.testing.assert_series_equal(result, expected)

    def test_sum_with_negative_values(self):
        """Test summing with negative values"""
        df = pd.DataFrame({
            'FZ1': [-100, 200, -300],
            'FZ2': [50, -75, 100]
        })

        result = sum_dual_force_components(df)

        expected = pd.Series([-50, 125, -200])
        pd.testing.assert_series_equal(result, expected)

    def test_sum_with_zero_values(self):
        """Test summing with zero values"""
        df = pd.DataFrame({
            'FZ1': [0, 0, 300],
            'FZ2': [0, 75, 0]
        })

        result = sum_dual_force_components(df)

        expected = pd.Series([0, 75, 300])
        pd.testing.assert_series_equal(result, expected)


class TestFindFirstFrameWhereForceExceedsThreshold:
    """Test find_first_frame_where_force_exceeds_threshold function"""

    def test_find_frame_normal_case(self):
        """Test finding frame in normal case"""
        force_trace = pd.Series([10, 20, 30, 100, 200, 300])
        threshold = 50

        result = find_first_frame_where_force_exceeds_threshold(force_trace, threshold)

        assert result == 3

    def test_find_frame_first_value_exceeds(self):
        """Test when first value exceeds threshold"""
        force_trace = pd.Series([100, 200, 300])
        threshold = 50

        result = find_first_frame_where_force_exceeds_threshold(force_trace, threshold)

        assert result == 0

    def test_find_frame_no_values_exceed(self):
        """Test when no values exceed threshold"""
        force_trace = pd.Series([10, 20, 30, 40])
        threshold = 50

        result = find_first_frame_where_force_exceeds_threshold(force_trace, threshold)

        assert result == -1

    def test_find_frame_exact_threshold(self):
        """Test when value exactly equals threshold"""
        force_trace = pd.Series([10, 20, 50, 60])
        threshold = 50

        result = find_first_frame_where_force_exceeds_threshold(force_trace, threshold)

        assert result == 2

    def test_find_frame_negative_values(self):
        """Test with negative force values"""
        force_trace = pd.Series([-100, -50, 0, 50, 100])
        threshold = 25

        result = find_first_frame_where_force_exceeds_threshold(force_trace, threshold)

        assert result == 3

    def test_find_frame_empty_series(self):
        """Test with empty series"""
        force_trace = pd.Series([])
        threshold = 50

        # ACTUAL BEHAVIOR: Function now raises ValueError for empty input
        with pytest.raises(ValueError, match="Force trace cannot be empty"):
            find_first_frame_where_force_exceeds_threshold(force_trace, threshold)


class TestFindFrameWhenOffPlate:
    """Test find_frame_when_off_plate function"""

    def test_find_takeoff_normal_case(self):
        """Test finding takeoff in normal case"""
        # Create force trace with clear takeoff (sustained low forces)
        force_trace = pd.Series([1000, 800, 600, 5, 5, 5, 5, 5, 400, 600])
        sampling_frequency = 1000

        result = find_frame_when_off_plate(force_trace, sampling_frequency, flight_time_threshold=0.003)

        assert result == 3

    def test_find_takeoff_no_sustained_low_force(self):
        """Test when no sustained low force period exists"""
        force_trace = pd.Series([1000, 800, 600, 5, 400, 600, 800])
        sampling_frequency = 1000

        result = find_frame_when_off_plate(force_trace, sampling_frequency)

        assert result == -100  # NOT_FOUND

    def test_find_takeoff_custom_thresholds(self):
        """Test with custom force and time thresholds"""
        force_trace = pd.Series([1000, 800, 600, 15, 15, 15, 15, 15, 400])
        sampling_frequency = 1000

        result = find_frame_when_off_plate(
            force_trace, sampling_frequency,
            flight_time_threshold=0.003,
            force_threshold=20
        )

        assert result == 3

    def test_find_takeoff_at_end_of_trace(self):
        """Test when takeoff occurs at end of trace"""
        force_trace = pd.Series([1000, 800, 600, 5, 5, 5])
        sampling_frequency = 1000

        result = find_frame_when_off_plate(force_trace, sampling_frequency, flight_time_threshold=0.002)

        assert result == 3

    def test_find_takeoff_negative_forces(self):
        """Test with negative force values (should be treated as low force)"""
        force_trace = pd.Series([1000, 800, -5, -5, -5, -5, 400])
        sampling_frequency = 1000

        result = find_frame_when_off_plate(force_trace, sampling_frequency, flight_time_threshold=0.002)

        assert result == 2


class TestGetNSecondsBeforeTakeoff:
    """Test get_n_seconds_before_takeoff function"""

    def test_crop_normal_case(self):
        """Test normal cropping case"""
        force_trace = pd.Series(range(100))  # 0 to 99
        sampling_frequency = 10  # 10 Hz
        takeoff_frame = 50
        n = 2  # 2 seconds = 20 samples

        result = get_n_seconds_before_takeoff(force_trace, sampling_frequency, takeoff_frame, n)

        expected = pd.Series(range(30, 50)).reset_index(drop=True)
        pd.testing.assert_series_equal(result, expected)

    def test_crop_near_beginning(self):
        """Test cropping when n seconds goes before start of trace"""
        force_trace = pd.Series(range(20))
        sampling_frequency = 10
        takeoff_frame = 5
        n = 2  # Would start at -15, should start at 0

        result = get_n_seconds_before_takeoff(force_trace, sampling_frequency, takeoff_frame, n)

        expected = pd.Series(range(0, 5)).reset_index(drop=True)
        pd.testing.assert_series_equal(result, expected)

    def test_crop_takeoff_frame_none(self):
        """Test when takeoff frame is None"""
        force_trace = pd.Series(range(100))
        sampling_frequency = 10
        takeoff_frame = None
        n = 2

        result = get_n_seconds_before_takeoff(force_trace, sampling_frequency, takeoff_frame, n)

        pd.testing.assert_series_equal(result, force_trace)

    def test_crop_different_n_values(self):
        """Test with different n values"""
        force_trace = pd.Series(range(100))
        sampling_frequency = 20  # 20 Hz
        takeoff_frame = 60

        # Test n = 1 second (20 samples)
        result_1s = get_n_seconds_before_takeoff(force_trace, sampling_frequency, takeoff_frame, 1)
        assert len(result_1s) == 20
        assert result_1s.iloc[0] == 40

        # Test n = 0.5 seconds (10 samples)
        result_half_s = get_n_seconds_before_takeoff(force_trace, sampling_frequency, takeoff_frame, 0.5)
        assert len(result_half_s) == 10
        assert result_half_s.iloc[0] == 50

    def test_crop_fractional_samples(self):
        """Test when n*sampling_frequency results in fractional samples"""
        force_trace = pd.Series(range(100))
        sampling_frequency = 13  # Odd frequency
        takeoff_frame = 50
        n = 1.5  # 1.5 * 13 = 19.5, should be truncated to 19

        result = get_n_seconds_before_takeoff(force_trace, sampling_frequency, takeoff_frame, n)

        assert len(result) == 19  # 50 - (50-19) = 19


class TestFindLandingFrame:
    """Test find_landing_frame function"""

    def test_find_landing_normal_case(self):
        """Test finding landing in normal case"""
        force_series = [5, 5, 25, 25, 25, 25, 10, 30, 30, 30]
        sampling_frequency = 1000

        result = find_landing_frame(force_series, sampling_frequency, time=0.003, threshold_value=20)

        assert result == 2

    def test_find_landing_no_sustained_contact(self):
        """Test when no sustained contact period exists"""
        force_series = [5, 25, 5, 25, 5, 25]
        sampling_frequency = 1000

        result = find_landing_frame(force_series, sampling_frequency, time=0.003, threshold_value=20)

        assert result == -1

    def test_find_landing_custom_thresholds(self):
        """Test with custom time and threshold values"""
        force_series = [5, 5, 50, 50, 50, 50, 50, 10]
        sampling_frequency = 1000

        result = find_landing_frame(
            force_series, sampling_frequency,
            time=0.005, threshold_value=40
        )

        assert result == 2

    def test_find_landing_at_beginning(self):
        """Test when landing occurs at beginning"""
        force_series = [25, 25, 25, 5, 5, 5]
        sampling_frequency = 1000

        result = find_landing_frame(force_series, sampling_frequency, time=0.002, threshold_value=20)

        assert result == 0

    def test_find_landing_exact_threshold(self):
        """Test when forces exactly equal threshold"""
        force_series = [5, 20, 20, 20, 20, 5]
        sampling_frequency = 1000

        result = find_landing_frame(force_series, sampling_frequency, time=0.003, threshold_value=20)

        assert result == 1

    def test_find_landing_short_series(self):
        """Test with series shorter than required time window"""
        force_series = [25, 25]
        sampling_frequency = 1000

        result = find_landing_frame(force_series, sampling_frequency, time=0.005, threshold_value=20)

        assert result == -1

    def test_find_landing_empty_series(self):
        """Test with empty series"""
        force_series = []
        sampling_frequency = 1000

        result = find_landing_frame(force_series, sampling_frequency, time=0.015, threshold_value=20)

        assert result == -1