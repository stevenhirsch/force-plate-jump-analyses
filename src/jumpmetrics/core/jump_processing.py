"""Wrapper function for generating jump metrics dataframe"""
import pandas as pd
import numpy as np
from jumpmetrics.core.processors import (
    ForceTimeCurveTakeoffProcessor,
    ForceTimeCurveCMJTakeoffProcessor,
    ForceTimeCurveSQJTakeoffProcessor,
    ForceTimeCurveJumpLandingProcessor
)
from jumpmetrics.core.io import (
    find_first_frame_where_force_exceeds_threshold,
    find_frame_when_off_plate, get_n_seconds_before_takeoff,
    find_landing_frame
)
from jumpmetrics.signal_processing.filters import butterworth_filter
from jumpmetrics.metrics.metrics import compute_jump_height_from_flight_time_events


def process_jump_trial(
    full_force_series, sampling_frequency: int, jump_type: str, weighing_time: float, pid: str,
    threshold_for_helping_determine_takeoff: float = 1000, seconds_before_takeoff: float = 2,
    seconds_for_determining_landing_phase: float = 0.015,
    lowpass_filter: bool = True, lowpass_cutoff_frequency: float = 50,
    compute_jump_height_from_flight_time: bool = False
) -> dict:
    """Wrapper function for processing a jump trial

    Args:
        full_force_series (array): The entire force series
        sampling_frequency (int): Sampling frequency of the force plate
        jump_type (str): Type of jump. Must be either "countermovement" or "squat"
        weighing_time (float): The time used to weigh the person before the jump
        pid (str): Participant ID (to append to result dataframe)
        threshold_for_helping_determine_takeoff (float, optional): Parameter for helping to determine takeoff.
        Defaults to 1000.
        seconds_before_takeoff (float, optional): Amount to crop data before takeoff. Defaults to 2.
        seconds_for_determining_landing_phase (float, optional): Amount of seconds for determining the landing phase.
        Defaults to 0.015 seconds.
        lowpass_filter (bool, optional): Whether to filter the force data or not. Defaults to True.
        lowpass_cutoff_frequency (float, optional): Lowpass filter cutoff frequency. Defaults to 50.

    Raises:
        ValueError: If invalid jump_type is specified

    Returns:
        dict: A dictionary of the resulting dataframe, takeoff force trace, and landing force trace.
    """
    force_series = pd.Series(full_force_series)
    valid_jump_types = ['countermovement', 'squat']
    if jump_type not in valid_jump_types:
        raise ValueError(f"jump_type must be in {valid_jump_types}. {jump_type} is not valid.")

    if lowpass_filter:
        force_series = butterworth_filter(
            arr=force_series,
            cutoff_frequency=lowpass_cutoff_frequency,
            fps=sampling_frequency,
            padding=sampling_frequency  # for 1sec padding
        )
        force_series = pd.Series(force_series)

    # Get Takeoff Trace
    frame_for_takeoff = find_first_frame_where_force_exceeds_threshold(
        force_trace=force_series,
        threshold=threshold_for_helping_determine_takeoff
    )
    takeoff_frame = find_frame_when_off_plate(
        force_trace=force_series[frame_for_takeoff:],
        sampling_frequency=sampling_frequency
    )
    takeoff_force_trace = get_n_seconds_before_takeoff(
        force_trace=force_series,
        sampling_frequency=sampling_frequency,
        takeoff_frame=takeoff_frame,
        n=seconds_before_takeoff
    )

    # Get Landing Trace
    force_series_after_takeoff = force_series[takeoff_frame:]
    landing_frame = find_landing_frame(
        force_series_after_takeoff,
        threshold_value=20,
        sampling_frequency=sampling_frequency,
        time=seconds_for_determining_landing_phase
    )
    off_forceplate_first_frame = find_frame_when_off_plate(
        force_trace=force_series_after_takeoff[landing_frame:],
        sampling_frequency=2000
    ) + landing_frame
    if off_forceplate_first_frame is not None:
        landing_force_trace = force_series_after_takeoff[landing_frame:off_forceplate_first_frame]
    else:
        landing_force_trace = force_series_after_takeoff[landing_frame:]

    # takeoff metrics
    takeoff: ForceTimeCurveTakeoffProcessor  # for typing only
    if jump_type == 'countermovement':
        takeoff = ForceTimeCurveCMJTakeoffProcessor(
            force_series=takeoff_force_trace,
            sampling_frequency=sampling_frequency,
            weighing_time=weighing_time
        )
    else:
        takeoff = ForceTimeCurveSQJTakeoffProcessor(
            force_series=takeoff_force_trace,
            sampling_frequency=sampling_frequency,
            weighing_time=weighing_time
        )

    takeoff.get_jump_events()
    takeoff.compute_jump_metrics()
    takeoff.create_jump_metrics_dataframe(
        pid=pid
    )

    # landing metrics
    landing = ForceTimeCurveJumpLandingProcessor(
        landing_force_trace=landing_force_trace,
        sampling_frequency=sampling_frequency,
        body_weight=takeoff.body_weight,
        takeoff_velocity=takeoff.jump_metrics['takeoff_velocity']
    )
    landing.get_landing_events()
    landing.compute_landing_metrics()
    landing.create_landing_metrics_dataframe(
        pid=pid
    )
    overall_dataframe = pd.merge(
        takeoff.jump_metrics_dataframe,
        landing.landing_metrics_dataframe,
        on='PID'
    )
    if compute_jump_height_from_flight_time:
        updated_landing_frame = takeoff_frame + landing_frame
        jump_height_flight_time = compute_jump_height_from_flight_time_events(
            takeoff_frame=takeoff_frame,
            landing_frame=updated_landing_frame,
            sampling_frequency=sampling_frequency
        )
        overall_dataframe['jump_height_flight_time'] = jump_height_flight_time
    else:
        overall_dataframe['jump_height_flight_time'] = np.nan
    overall_dataframe['flight_time'] = (updated_landing_frame - takeoff_frame) / sampling_frequency

    results_dict = {
        'results_dataframe': overall_dataframe,
        'takeoff_force_trace': takeoff_force_trace,
        'landing_force_trace': landing_force_trace.reset_index(drop=True)
    }
    return results_dict
