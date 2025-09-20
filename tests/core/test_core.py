"""Tests for core.py"""
import os
import numpy as np
import pandas as pd
from jumpmetrics.core.io import load_cropped_force_data
from jumpmetrics.signal_processing.filters import butterworth_filter
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
