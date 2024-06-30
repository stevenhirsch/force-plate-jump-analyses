"""Batch processing script for study 2 - WIP, just plotting for now"""
import os
import matplotlib.pyplot as plt
# from jumpmetrics.core.core import ForceTimeCurveCMJTakeoffProcessor
from jumpmetrics.core.io import (
    load_raw_force_data, sum_dual_force_components, find_first_frame_where_force_exceeds_threshold,
    find_takeoff_frame, get_n_seconds_before_takeoff
)
# from jumpmetrics.signal_processing.filters import butterworth_filter
# import pandas as pd
# from tqdm import tqdm


main_dir = os.path.join(os.getcwd(), 'analyses', 'study_2')
sample_filepath = os.path.join(main_dir, 'F02_CTRL1_converted.txt')

df = load_raw_force_data(sample_filepath)
full_summed_force = sum_dual_force_components(df)
frame = find_first_frame_where_force_exceeds_threshold(
    force_trace=full_summed_force,
    threshold=1000
)
takeoff_frame = find_takeoff_frame(
    force_trace=full_summed_force.iloc[frame:],
    sampling_frequency=2000
)
cropped_force_trace = get_n_seconds_before_takeoff(
    force_trace=full_summed_force,
    sampling_frequency=2000,
    takeoff_frame=takeoff_frame,
    n=2
)
plt.plot(cropped_force_trace.index / 2000, cropped_force_trace)
plt.xlabel('Time')
plt.ylabel('Force (N)')
plt.title('Example Cropped Force-Time Trace for Just Takeoff')
plt.show()
