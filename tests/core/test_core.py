"""Tests for core.py"""
import os
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from jumpmetrics.core.io import load_cropped_force_data
from jumpmetrics.events.cmj_events import NOT_FOUND
from jumpmetrics.signal_processing.filters import butterworth_filter
from jumpmetrics.signal_processing.numerical import integrate_area
from jumpmetrics.core.processors import (
    ForceTimeCurveCMJTakeoffProcessor, ForceTimeCurveSQJTakeoffProcessor, ForceTimeCurveJumpLandingProcessor
)
from jumpmetrics.core.jump_processing import process_jump_trial
from jumpmetrics.core.io import(
    load_raw_force_data_with_no_column_headers, sum_dual_force_components,
    find_first_frame_where_force_exceeds_threshold,
    find_frame_when_off_plate, get_n_seconds_before_takeoff
)

ERROR_THRESHOLD = 1e-3
main_dir = os.path.join('tests', 'example_data')
data_dir = os.path.join(main_dir, 'raw_data')
kinematic_data_dir = os.path.join(main_dir, 'kinematic_data')
batch_processed_results = pd.read_csv(main_dir + '/batch_processed_data.csv')

FILTER_TYPE = 'group_cutoff'
GROUP_CUTOFF_FREQUENCY = 26.648
# Integration Test 1
def test_ForceTimeCurveCMJTakeoffProcessor_1():
    """Basic integration test"""
    testfile = os.path.join(
        data_dir, 'F02_CTRL2' + '_filtered.txt'
    )
    testresults_kinematics_path = os.path.join(
        kinematic_data_dir, 'F02', 'CTRL2', FILTER_TYPE + '.csv'
    )
    testreults_kinetics_path = os.path.join(
        kinematic_data_dir, 'F02', 'CTRL2', FILTER_TYPE + '_force_series.csv'
    )

    testresults_kinematics = pd.read_csv(testresults_kinematics_path)
    testresults_acceleration = testresults_kinematics.acceleration
    testresults_velocity = testresults_kinematics.velocity
    testresults_displacement = testresults_kinematics.displacement
    testresults_kinetics = pd.read_csv(testreults_kinetics_path)
    testresults_force = testresults_kinetics.force

    force_series = load_cropped_force_data(
        filepath=testfile,
        freq=None
    )
    filtered_force_series = butterworth_filter(
        arr=force_series,
        cutoff_frequency=GROUP_CUTOFF_FREQUENCY,
        fps=2000,
        padding=2000
    )
    CMJ = ForceTimeCurveCMJTakeoffProcessor(
        # force_series=filtered_force_series,
        force_series=filtered_force_series[-4000:],
        sampling_frequency=2000,
        weighing_time=0.25
    )
    CMJ.get_jump_events()
    CMJ.compute_jump_metrics()
    CMJ.create_jump_metrics_dataframe(pid='F02')
    CMJ.create_kinematic_dataframe()

    kinematics_dataframe = batch_processed_results[
        (batch_processed_results.file_prefix == 'F02_CTRL2') &
        (batch_processed_results.cutoff_type == 'group_cutoff')
    ].drop(['file_prefix', 'cutoff_type', 'cutoff_frequency'], axis=1).reset_index(drop=True)
    assert len(CMJ.force_series) == len(testresults_force)
    assert np.allclose(CMJ.force_series, testresults_force, equal_nan=True)
    assert np.allclose(CMJ.acceleration_series, testresults_acceleration)
    assert np.allclose(CMJ.velocity_series, testresults_velocity)
    assert np.allclose(CMJ.displacement_series, testresults_displacement)
    assert np.all(kinematics_dataframe.dtypes == CMJ.jump_metrics_dataframe.dtypes)
    assert kinematics_dataframe.index.equals(CMJ.jump_metrics_dataframe.index)
    assert kinematics_dataframe.columns.equals(CMJ.jump_metrics_dataframe.columns)
    diffs = np.abs(
        kinematics_dataframe.drop('PID', axis=1).values -
        CMJ.jump_metrics_dataframe.drop('PID', axis=1).values
    )[0]
    assert np.all(diffs < ERROR_THRESHOLD)

# Integration Test 2
def test_ForceTimeCurveCMJTakeoffProcessor_2():
    """Basic integration test"""
    testfile = os.path.join(
        data_dir, 'M07_CTRL1' + '_filtered.txt'
    )
    testresults_kinematics_path = os.path.join(
        kinematic_data_dir, 'M07', 'CTRL1', FILTER_TYPE + '.csv'
    )
    testreults_kinetics_path = os.path.join(
        kinematic_data_dir, 'M07', 'CTRL1', FILTER_TYPE + '_force_series.csv'
    )

    testresults_kinematics = pd.read_csv(testresults_kinematics_path)
    testresults_acceleration = testresults_kinematics.acceleration
    testresults_velocity = testresults_kinematics.velocity
    testresults_displacement = testresults_kinematics.displacement
    testresults_kinetics = pd.read_csv(testreults_kinetics_path)
    testresults_force = testresults_kinetics.force

    force_series = load_cropped_force_data(
        filepath=testfile,
        freq=None
    )
    filtered_force_series = butterworth_filter(
        arr=force_series,
        cutoff_frequency=GROUP_CUTOFF_FREQUENCY,
        fps=2000,
        padding=2000
    )
    CMJ = ForceTimeCurveCMJTakeoffProcessor(
        # force_series=filtered_force_series,
        force_series=filtered_force_series[-4000:],
        sampling_frequency=2000
    )
    CMJ.get_jump_events()
    CMJ.compute_jump_metrics()
    CMJ.create_jump_metrics_dataframe(pid='M07')
    CMJ.create_kinematic_dataframe()

    kinematics_dataframe = batch_processed_results[
        (batch_processed_results.file_prefix == 'M07_CTRL1') &
        (batch_processed_results.cutoff_type == 'group_cutoff')
    ].drop(['file_prefix', 'cutoff_type', 'cutoff_frequency'], axis=1).reset_index(drop=True)
    assert len(CMJ.force_series) == len(testresults_force)
    assert np.allclose(CMJ.force_series, testresults_force, equal_nan=True)
    assert np.allclose(CMJ.acceleration_series, testresults_acceleration)
    assert np.allclose(CMJ.velocity_series, testresults_velocity)
    assert np.allclose(CMJ.displacement_series, testresults_displacement)
    assert np.all(kinematics_dataframe.dtypes == CMJ.jump_metrics_dataframe.dtypes)
    assert kinematics_dataframe.index.equals(CMJ.jump_metrics_dataframe.index)
    assert kinematics_dataframe.columns.equals(CMJ.jump_metrics_dataframe.columns)
    diffs = np.abs(
        kinematics_dataframe.drop('PID', axis=1).values -
        CMJ.jump_metrics_dataframe.drop('PID', axis=1).values
    )[0]
    assert np.all(diffs < ERROR_THRESHOLD)

# Integration Test 3
def test_ForceTimeCurveCMJTakeoffProcessor_3():
    """Basic integration test"""
    testfile = os.path.join(
        data_dir, 'M15_CTRL1' + '_filtered.txt'
    )
    testresults_kinematics_path = os.path.join(
        kinematic_data_dir, 'M15', 'CTRL1', FILTER_TYPE + '.csv'
    )
    testreults_kinetics_path = os.path.join(
        kinematic_data_dir, 'M15', 'CTRL1', FILTER_TYPE + '_force_series.csv'
    )

    testresults_kinematics = pd.read_csv(testresults_kinematics_path)
    testresults_acceleration = testresults_kinematics.acceleration
    testresults_velocity = testresults_kinematics.velocity
    testresults_displacement = testresults_kinematics.displacement
    testresults_kinetics = pd.read_csv(testreults_kinetics_path)
    testresults_force = testresults_kinetics.force

    force_series = load_cropped_force_data(
        filepath=testfile,
        freq=None
    )
    filtered_force_series = butterworth_filter(
        arr=force_series,
        cutoff_frequency=GROUP_CUTOFF_FREQUENCY,
        fps=2000,
        padding=2000
    )
    CMJ = ForceTimeCurveCMJTakeoffProcessor(
        # force_series=filtered_force_series,
        force_series=filtered_force_series[-4000:],
        sampling_frequency=2000
    )
    CMJ.get_jump_events()
    CMJ.compute_jump_metrics()
    CMJ.create_jump_metrics_dataframe(pid='M15')
    CMJ.create_kinematic_dataframe()

    kinematics_dataframe = batch_processed_results[
        (batch_processed_results.file_prefix == 'M15_CTRL1') &
        (batch_processed_results.cutoff_type == 'group_cutoff')
    ].drop(['file_prefix', 'cutoff_type', 'cutoff_frequency'], axis=1).reset_index(drop=True)
    assert len(CMJ.force_series) == len(testresults_force)
    assert np.allclose(CMJ.force_series, testresults_force, equal_nan=True)
    assert np.allclose(CMJ.acceleration_series, testresults_acceleration)
    assert np.allclose(CMJ.velocity_series, testresults_velocity)
    assert np.allclose(CMJ.displacement_series, testresults_displacement)
    assert np.all(kinematics_dataframe.dtypes == CMJ.jump_metrics_dataframe.dtypes)
    assert kinematics_dataframe.index.equals(CMJ.jump_metrics_dataframe.index)
    assert kinematics_dataframe.columns.equals(CMJ.jump_metrics_dataframe.columns)
    diffs = np.abs(
        kinematics_dataframe.drop('PID', axis=1).values -
        CMJ.jump_metrics_dataframe.drop('PID', axis=1).values
    )[0]
    assert np.all(diffs < ERROR_THRESHOLD)


IMPULSE_WINDOW_TEST_TRIALS = ['F02_CTRL2', 'M07_CTRL1', 'M15_CTRL1']
IMPULSE_RELATIVE_TOLERANCE = 0.01


def _process_cmj_trial_for_impulse_checks(file_prefix: str) -> ForceTimeCurveCMJTakeoffProcessor:
    """Helper to reprocess one of the bundled CMJ fixtures for the impulse-window tests below"""
    testfile = os.path.join(data_dir, file_prefix + '_filtered.txt')
    force_series = load_cropped_force_data(filepath=testfile, freq=None)
    filtered_force_series = butterworth_filter(
        arr=force_series,
        cutoff_frequency=GROUP_CUTOFF_FREQUENCY,
        fps=2000,
        padding=2000
    )
    CMJ = ForceTimeCurveCMJTakeoffProcessor(
        force_series=filtered_force_series[-4000:],
        sampling_frequency=2000
    )
    CMJ.get_jump_events()
    CMJ.compute_jump_metrics()
    return CMJ


def test_impulse_windows_are_additive_and_match_momentum_identities():
    """The three net-vertical-impulse windows must partition the trial without gaps or overlaps
    (braking + propulsive == braking-to-propulsive, exactly, since all three are trapezoidal
    integrals of the same signal), and each impulse must equal the momentum change over its own
    window (impulse-momentum theorem), since start_of_propulsive_phase is the velocity
    zero-crossing (the low position), so velocity there is ~0 and propulsive_net_vertical_impulse
    ~= body_mass_kg * takeoff_velocity.
    """
    for file_prefix in IMPULSE_WINDOW_TEST_TRIALS:
        CMJ = _process_cmj_trial_for_impulse_checks(file_prefix)
        body_mass_kg = CMJ.body_mass_kg
        velocity = CMJ.velocity_series
        start_of_braking_phase = CMJ.start_of_braking_phase
        start_of_propulsive_phase = CMJ.start_of_propulsive_phase

        braking_nvi = CMJ.jump_metrics['braking_net_vertical_impulse']
        propulsive_nvi = CMJ.jump_metrics['propulsive_net_vertical_impulse']
        braking_to_propulsive_nvi = CMJ.jump_metrics['braking_to_propulsive_net_vertical_impulse']
        total_nvi = CMJ.jump_metrics['total_net_vertical_impulse']
        takeoff_velocity = velocity[-1]

        # Additivity: the two phase windows share their boundary sample, so they must sum exactly
        # to the combined braking-to-propulsive window (same integrator, same signal).
        assert braking_nvi + propulsive_nvi == pytest.approx(braking_to_propulsive_nvi, abs=1e-9), (
            f'{file_prefix}: braking_nvi + propulsive_nvi != braking_to_propulsive_nvi'
        )

        # Impulse-momentum theorem, per window.
        expected_braking_nvi = body_mass_kg * (
            velocity[start_of_propulsive_phase] - velocity[start_of_braking_phase]
        )
        assert braking_nvi == pytest.approx(expected_braking_nvi, rel=IMPULSE_RELATIVE_TOLERANCE), (
            f'{file_prefix}: braking_nvi does not match body_mass_kg * delta-v over the braking window'
        )

        expected_propulsive_nvi = body_mass_kg * takeoff_velocity
        assert propulsive_nvi == pytest.approx(expected_propulsive_nvi, rel=IMPULSE_RELATIVE_TOLERANCE), (
            f'{file_prefix}: propulsive_nvi does not match body_mass_kg * takeoff_velocity'
        )

        expected_total_nvi = body_mass_kg * takeoff_velocity
        assert total_nvi == pytest.approx(expected_total_nvi, rel=IMPULSE_RELATIVE_TOLERANCE), (
            f'{file_prefix}: total_net_vertical_impulse does not match body_mass_kg * takeoff_velocity'
        )


def test_impulse_window_additivity_catches_a_wrong_boundary():
    """Regression guard for the additivity check itself: if the braking window no longer shares its
    boundary sample with the propulsive window (e.g. `low_position_end` loses its `+1`), the two
    phase impulses must stop summing exactly to the combined window, and the additivity assertion
    above must be able to catch it.
    """
    CMJ = _process_cmj_trial_for_impulse_checks('F02_CTRL2')
    start_of_propulsive_phase = CMJ.start_of_propulsive_phase
    n_frames = len(CMJ.force_series)

    # Reproduce the pre-fix, non-shared-boundary windows directly, bypassing the processor.
    broken_braking_nvi = integrate_area(
        time=CMJ.time[CMJ.start_of_braking_phase:start_of_propulsive_phase],
        signal=CMJ.force_series_minus_bodyweight[CMJ.start_of_braking_phase:start_of_propulsive_phase]
    )
    broken_propulsive_nvi = integrate_area(
        time=CMJ.time[start_of_propulsive_phase:n_frames],
        signal=CMJ.force_series_minus_bodyweight[start_of_propulsive_phase:n_frames]
    )
    broken_sum = broken_braking_nvi + broken_propulsive_nvi
    braking_to_propulsive_nvi = CMJ.jump_metrics['braking_to_propulsive_net_vertical_impulse']

    assert broken_sum != pytest.approx(braking_to_propulsive_nvi, abs=1e-9), (
        'Expected dropping the shared boundary sample to break additivity, but it still summed exactly '
        '- the additivity assertion would not have caught this regression'
    )


def test_plot_specific_events_skips_peak_force_when_not_found():
    """`get_peak_force_event` can now return NOT_FOUND (-100) instead of always falling back to
    argmax, so `_plot_specific_events` must not draw a 'Peak Force' line for that sentinel.
    """
    CMJ = _process_cmj_trial_for_impulse_checks('F02_CTRL2')
    CMJ.peak_force_frame = NOT_FOUND

    plt.figure()
    CMJ._plot_specific_events()  # pylint: disable=protected-access
    line_labels = [line.get_label() for line in plt.gca().get_lines()]
    plt.close()

    assert 'Peak Force' not in line_labels


def test_peak_force_detects_bimodal_pre_low_position_peak():
    """F04_CTRL2 is a real trial with a genuinely bimodal force-time curve: the true peak occurs
    well before the low position, exactly the shape from the original bug report (issue #2). This
    pins the fix: `peak_force_frame` must be found by the braking-anchored search, i.e. it must fall
    before `start_of_propulsive_phase` -- a frame the old propulsive-anchored search could never
    have returned, since it never looked there.
    """
    CMJ = _process_cmj_trial_for_impulse_checks('F04_CTRL2')

    # The true peak is before the low position - only reachable by a braking-anchored search.
    assert CMJ.peak_force_frame < CMJ.start_of_propulsive_phase
    assert not np.isnan(CMJ.jump_metrics['peak_force'])


def test_propulsive_peak_force_not_found_on_monotonic_propulsive_phase():
    """M06_CTRL3 is a real trial with no additional prominent peak between the low position and
    takeoff, so the propulsive-anchored search (used only for the propulsive_peakforce_rfd_*
    metrics) legitimately finds nothing. This exercises the `NOT_FOUND` path end-to-end: previously
    masked by an `np.argmax` fallback, `frame_propulsive_peak_force` must now be `NOT_FOUND` and the
    three RFD metrics that depend on it must be `np.nan`, while `peak_force` itself (found by the
    separate, braking-anchored search) must remain valid.
    """
    CMJ = _process_cmj_trial_for_impulse_checks('M06_CTRL3')

    assert CMJ.propulsive_peak_force_frame == NOT_FOUND
    assert not np.isnan(CMJ.jump_metrics['peak_force'])
    for metric in [
        'propulsive_peakforce_rfd_slope_between_events',
        'propulsive_peakforce_rfd_instantaneous_average_between_events',
        'propulsive_peakforce_rfd_instantaneous_peak_between_events',
    ]:
        assert np.isnan(CMJ.jump_metrics[metric]), f'{metric} should be NaN when the propulsive peak is not found'


# Integration Test 4
def test_process_jump_data_wrapper_func_1():
    """Basic integration test for wrapper function"""
    results_df = pd.read_csv(main_dir + '/example_process_jump_trial_output.csv')
    test_filepath = os.path.join(data_dir, 'F02_CTRL1.txt')
    tmp_force_df = load_raw_force_data_with_no_column_headers(test_filepath)
    full_summed_force = sum_dual_force_components(tmp_force_df)
    results_dict = process_jump_trial(
        full_force_series=full_summed_force,
        sampling_frequency=2000,
        jump_type='countermovement',
        weighing_time=0.25,
        pid='F02',
        threshold_for_helping_determine_takeoff=1000,
        lowpass_filter=True,
        lowpass_cutoff_frequency=26.64,
        compute_jump_height_from_flight_time=True
    )
    df = results_dict['results_dataframe']
    diffs = np.abs(
        df.drop('PID', axis=1).values -
        results_df.drop('PID', axis=1).values
    )[0]
    assert np.all(diffs < ERROR_THRESHOLD)

# Integration Test 5
def test_ForceTimeCurveSQJTakeoffProcessor_1():
    """Basic integration test for squat jump takeoff processor"""
    results_df = pd.read_csv(main_dir + '/p30_squat_jump_example_results.csv')
    test_filepath = os.path.join(data_dir, 'FHOC_P30_SQT300007.txt')
    tmp_force_df = load_raw_force_data_with_no_column_headers(test_filepath)
    full_summed_force = sum_dual_force_components(tmp_force_df)
    cutoff_frequency = 50
    sampling_frequency = 1000
    TIME_BEFORE_TAKEOFF = 2
    THRESHOLD = 1000
    frame = find_first_frame_where_force_exceeds_threshold(
        force_trace=full_summed_force,
        threshold=THRESHOLD
    )
    takeoff_frame = find_frame_when_off_plate(
        force_trace=full_summed_force.iloc[frame:],
        sampling_frequency=sampling_frequency
    )
    cropped_force_trace = get_n_seconds_before_takeoff(
        force_trace=full_summed_force,
        sampling_frequency=sampling_frequency,
        takeoff_frame=takeoff_frame,
        n=TIME_BEFORE_TAKEOFF
    )
    filtered_force_series = butterworth_filter(
        arr=cropped_force_trace,
        cutoff_frequency=cutoff_frequency,
        fps=sampling_frequency,
        padding=sampling_frequency
    )
    sqj = ForceTimeCurveSQJTakeoffProcessor(
        force_series=filtered_force_series,
        sampling_frequency=sampling_frequency,
        weighing_time=0.5
    )
    sqj.get_jump_events(
        threshold_factor_for_propulsion=10,
        threshold_factor_for_unweighting=3
    )
    sqj.compute_jump_metrics()

    sqj.create_kinematic_dataframe()

    sqj.create_jump_metrics_dataframe(
        pid='P30'
    )
    diffs = np.abs(
        sqj.jump_metrics_dataframe.drop('PID', axis=1).values -
        results_df.drop('PID', axis=1).values
    )[0]
    assert np.all(diffs < ERROR_THRESHOLD)

# Integration Test 6
def test_ForceTimeCurveSQJTakeoffProcessor_2():
    """Basic integration test for squat jump takeoff processor"""
    results_df = pd.read_csv(main_dir + '/p31_squat_jump_example_results.csv')
    test_filepath = os.path.join(data_dir, 'FHOC_P31_SQT900020.txt')
    tmp_force_df = load_raw_force_data_with_no_column_headers(test_filepath)
    full_summed_force = sum_dual_force_components(tmp_force_df)
    cutoff_frequency = 50
    sampling_frequency = 1000
    TIME_BEFORE_TAKEOFF = 2
    THRESHOLD = 1000
    frame = find_first_frame_where_force_exceeds_threshold(
        force_trace=full_summed_force,
        threshold=THRESHOLD
    )
    takeoff_frame = find_frame_when_off_plate(
        force_trace=full_summed_force.iloc[frame:],
        sampling_frequency=sampling_frequency
    )
    cropped_force_trace = get_n_seconds_before_takeoff(
        force_trace=full_summed_force,
        sampling_frequency=sampling_frequency,
        takeoff_frame=takeoff_frame,
        n=TIME_BEFORE_TAKEOFF
    )
    filtered_force_series = butterworth_filter(
        arr=cropped_force_trace,
        cutoff_frequency=cutoff_frequency,
        fps=sampling_frequency,
        padding=sampling_frequency
    )
    sqj = ForceTimeCurveSQJTakeoffProcessor(
        force_series=filtered_force_series,
        sampling_frequency=sampling_frequency,
        weighing_time=0.5
    )
    sqj.get_jump_events(
        threshold_factor_for_propulsion=10,
        threshold_factor_for_unweighting=3
    )
    sqj.compute_jump_metrics()

    sqj.create_kinematic_dataframe()

    sqj.create_jump_metrics_dataframe(
        pid='P31'
    )
    diffs = np.abs(
        sqj.jump_metrics_dataframe.drop('PID', axis=1).values -
        results_df.drop('PID', axis=1).values
    )[0]
    diffs_less_than_threshold = np.all(diffs < ERROR_THRESHOLD)
    if not diffs_less_than_threshold:
        print(diffs)
    assert diffs_less_than_threshold

# Integration Test 7
def test_process_jump_data_wrapper_func_2():
    """Basic integration test for wrapper function"""
    results_df = pd.read_csv(main_dir + '/example_process_squat_jump_trial_output.csv')
    test_filepath = os.path.join(data_dir, 'FHOC_P30_SQT300007.txt')
    tmp_force_df = load_raw_force_data_with_no_column_headers(test_filepath)
    full_summed_force = sum_dual_force_components(tmp_force_df)
    results_dict = process_jump_trial(
        full_force_series=full_summed_force,
        sampling_frequency=1000,
        jump_type='squat',
        weighing_time=0.5,
        pid='P30',
        threshold_for_helping_determine_takeoff=1000,
        seconds_for_determining_landing_phase=0.030,
        lowpass_filter=True,
        lowpass_cutoff_frequency=26.64,
        compute_jump_height_from_flight_time=True
    )
    df = results_dict['results_dataframe']
    diffs = np.abs(
        df.drop('PID', axis=1).values -
        results_df.drop('PID', axis=1).values
    )[0]
    assert np.all(diffs < ERROR_THRESHOLD)


# Integration Test 8
def test_random_input_data():
    """Testing classes with random input data"""
    np.random.seed(0)
    random_data = np.random.randn(2200) * 750  # 1.1s at 2000Hz - above minimum requirement
    random_bw = np.random.rand(1)[0]
    random_takeoff_vel = np.random.rand(1)[0]

    sqj = ForceTimeCurveSQJTakeoffProcessor(
        force_series=random_data,
        sampling_frequency=2000
    )
    sqj.get_jump_events(
        threshold_factor_for_propulsion=10,
        threshold_factor_for_unweighting=3
    )
    sqj.compute_jump_metrics()

    sqj.create_kinematic_dataframe()

    sqj.create_jump_metrics_dataframe(
        pid='test'
    )

    cmj = ForceTimeCurveCMJTakeoffProcessor(
        force_series=random_data,
        sampling_frequency=2000
    )
    cmj.get_jump_events()
    cmj.compute_jump_metrics()

    cmj.create_kinematic_dataframe()

    cmj.create_jump_metrics_dataframe(
        pid='test'
    )

    landing = ForceTimeCurveJumpLandingProcessor(
        landing_force_trace=random_data,
        sampling_frequency=2000,
        body_weight=random_bw,
        takeoff_velocity=random_takeoff_vel
    )
    landing.get_landing_events()
    landing.compute_landing_metrics()
    landing.create_landing_metrics_dataframe(
        pid='test'
    )
    # Not asserting anything, just want to make sure that
    # nothing crashes with random data
