"""Example script demonstrating how to use jumpmetrics in Docker"""
import os
import pandas as pd

from jumpmetrics.core.processors import ForceTimeCurveCMJTakeoffProcessor
from jumpmetrics.core.io import (
    load_raw_force_data_with_no_column_headers, sum_dual_force_components,
    find_first_frame_where_force_exceeds_threshold,
    find_frame_when_off_plate, get_n_seconds_before_takeoff
)
from jumpmetrics.signal_processing.filters import butterworth_filter

def process_jump_trial():
    # Load example data
    filepath = '/data/input/F02_CTRL1.txt'
    tmp_force_df = load_raw_force_data_with_no_column_headers(filepath)
    full_summed_force = sum_dual_force_components(tmp_force_df)
    # Filter the full signal first to avoid border effects on the cropped window
    # butterworth_filter returns a numpy array, so wrap in pd.Series for downstream functions
    cutoff_frequency = 50
    filtered_full_force = pd.Series(butterworth_filter(
        arr=full_summed_force,
        cutoff_frequency=cutoff_frequency,
        fps=2000,
        padding=2000
    ))
    frame = find_first_frame_where_force_exceeds_threshold(
        force_trace=filtered_full_force,
        threshold=1000
    )
    takeoff_frame = find_frame_when_off_plate(
        force_trace=filtered_full_force.iloc[frame:],
        sampling_frequency=2000
    )
    TIME_BEFORE_TAKEOFF = 2
    cropped_force_trace = get_n_seconds_before_takeoff(
        force_trace=filtered_full_force,
        sampling_frequency=2000,
        takeoff_frame=takeoff_frame,
        n=TIME_BEFORE_TAKEOFF
    )

    # Initialize processor
    processor = ForceTimeCurveCMJTakeoffProcessor(
        force_series=cropped_force_trace,
        sampling_frequency=2000,  # Standard sampling frequency
        weighing_time=0.4  # Time window for computing body weight
    )
    
    # Get jump events
    processor.get_jump_events()
    
    # Compute metrics
    processor.compute_jump_metrics()
    
    # Create metrics dataframe
    processor.create_jump_metrics_dataframe(pid='F02_CTRL1')
    
    # Save results
    output_dir = '/data/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    processor.save_jump_metrics_dataframe(os.path.join(output_dir, 'jump_metrics.csv'))
    
    # Create and save kinematic data
    processor.create_kinematic_dataframe()
    processor.save_kinematic_dataframe(os.path.join(output_dir, 'kinematic_data.csv'))
    
    # Generate plots
    for waveform in ['force', 'velocity', 'displacement']:
        processor.plot_waveform(
            waveform_type=waveform,
            title=f'F02_CTRL1 {waveform.capitalize()} Curve',
            savefig=True,
            figname=os.path.join(output_dir, f'{waveform}_curve.png')
        )

if __name__ == '__main__':
    process_jump_trial() 