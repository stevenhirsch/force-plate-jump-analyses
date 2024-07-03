"""Batch processing script for study 2 - WIP, just plotting for now"""
import os
import logging
import pandas as pd
from tqdm import tqdm
from jumpmetrics.core.core import ForceTimeCurveCMJTakeoffProcessor
from jumpmetrics.core.io import (
    load_raw_force_data_with_no_column_headers, sum_dual_force_components,
    find_first_frame_where_force_exceeds_threshold,
    find_takeoff_frame, get_n_seconds_before_takeoff
)
from jumpmetrics.signal_processing.filters import butterworth_filter

class WarningCatcher(logging.Handler):
    def __init__(self):
        super().__init__()
        self.warnings = []

    def emit(self, record):
        if record.levelno == logging.WARNING:
            self.warnings.append(record.getMessage())

    def reset(self):
        self.warnings = []

# Set up the warning catcher
warning_catcher = WarningCatcher()
logging.getLogger().addHandler(warning_catcher)

main_dir = os.path.join(os.getcwd(), 'analyses', 'study_2')
data_dir = os.path.join(main_dir, 'raw_data')
fig_dir = os.path.join(main_dir, 'figures')
all_filenames = os.listdir(data_dir)
xlsx_data_filepath = os.path.join(main_dir, 'signal_conditioning_filter_cutoffs.xlsx')
xlsx_data = pd.read_excel(xlsx_data_filepath)
full_dataset = pd.DataFrame()
xlsx_data['pid'] = xlsx_data['file_prefix'].str.split('_').str[0]
xlsx_data_condensed = xlsx_data.groupby('pid').mean(numeric_only=True)
xlsx_data_condensed.drop(['trial_cutoff', 'participant_cutoff', 'literature_cutoff'], axis=1, inplace=True)
xlsx_data_condensed.reset_index(inplace=True)

for filename in tqdm(all_filenames):
    warning_catcher.reset()
    filepath = os.path.join(data_dir, filename)
    if '.DS_Store' in filepath:
        continue
    FILE_PREFIX = filename.split('.')[0]
    pid = FILE_PREFIX.split('_', maxsplit=1)[0]
    trial_num = FILE_PREFIX.split('_')[1]
    pid_fig_dir = os.path.join(fig_dir, pid, trial_num)
    if not os.path.exists(pid_fig_dir):
        os.makedirs(pid_fig_dir)
    participant_filters = xlsx_data_condensed[xlsx_data_condensed['pid'] == pid]
    cutoff_frequency = participant_filters['group_cutoff'].item()
    try:
        tmp_df = pd.DataFrame()
        tmp_df['file_prefix'] = [FILE_PREFIX]
        tmp_df['cutoff_type'] = ['group_cutoff']
        cutoff_frequency = participant_filters['group_cutoff'].item()
        tmp_df['cutoff_frequency'] = [cutoff_frequency]
        tmp_force_df = load_raw_force_data_with_no_column_headers(filepath)
        full_summed_force = sum_dual_force_components(tmp_force_df)
        frame = find_first_frame_where_force_exceeds_threshold(
            force_trace=full_summed_force,
            threshold=1000
        )
        takeoff_frame = find_takeoff_frame(
            force_trace=full_summed_force.iloc[frame:],
            sampling_frequency=2000
        )
        if pid in ['M05', 'F06']:
            TIME_BEFORE_TAKEOFF = 3
        else:
            TIME_BEFORE_TAKEOFF = 2
        cropped_force_trace = get_n_seconds_before_takeoff(
            force_trace=full_summed_force,
            sampling_frequency=2000,
            takeoff_frame=takeoff_frame,
            n=TIME_BEFORE_TAKEOFF
        )
        filtered_force_series = butterworth_filter(
            arr=cropped_force_trace,
            cutoff_frequency=cutoff_frequency,
            fps=2000,
            padding=2000
        )
        CMJ = ForceTimeCurveCMJTakeoffProcessor(
            # force_series=filtered_force_series,
            force_series=filtered_force_series,
            sampling_frequency=2000
        )
        CMJ.get_jump_events()
        CMJ.compute_jump_metrics()
        CMJ.create_jump_metrics_dataframe(pid=pid)
        CMJ.create_kinematic_dataframe()
        CMJ.plot_waveform(
            waveform_type='force',
            title=filename + '_group_cutoff',
            savefig=True,
            figname=os.path.join(pid_fig_dir, 'group_cutoff.png')
        )
        tmp_df = pd.concat([
            tmp_df, CMJ.jump_metrics_dataframe
        ], axis=1)
        full_dataset = pd.concat([
            full_dataset, tmp_df
        ], axis=0)
    except ValueError as e:
        print(f'Skipping {filename} due to ValueError')
        print('')
        print(f"Specific ValueErrror: {e}")
        CMJ.plot_waveform(waveform_type='force', title=filename + '_group_cutoff')
    if warning_catcher.warnings:
        print(f"Warnings occurred while processing {filename}:")
        for warning in warning_catcher.warnings:
            print(f"  - {warning}")

processed_file_name = os.path.join(main_dir, 'batch_processed_data.csv')
print('')
print('Saving results...')
full_dataset = full_dataset.sort_values(by='file_prefix')
split_prefix = full_dataset['file_prefix'].str.split('_', expand=True)
trial_metadata = split_prefix[1]
full_dataset['trial_type'] = trial_metadata.str[:-1]
full_dataset['trial_number'] = trial_metadata.str[-1]
full_dataset.to_csv(processed_file_name, index=False)
