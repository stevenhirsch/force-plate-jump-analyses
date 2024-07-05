"""Tests for core.py"""
import os
import numpy as np
import pandas as pd
from jumpmetrics.core.io import load_cropped_force_data
from jumpmetrics.signal_processing.filters import butterworth_filter
from jumpmetrics.core.core import ForceTimeCurveCMJTakeoffProcessor


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
        sampling_frequency=2000
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
    assert np.all(diffs < 1e-3)

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
    assert np.all(diffs < 1e-3)

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
    assert np.all(diffs < 1e-3)
