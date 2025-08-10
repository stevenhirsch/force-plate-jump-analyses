"""Batch processing script for study 2"""
import os
import logging
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from jumpmetrics.core.processors import ForceTimeCurveCMJTakeoffProcessor
from jumpmetrics.core.io import (
    load_raw_force_data_with_no_column_headers, sum_dual_force_components,
    find_first_frame_where_force_exceeds_threshold,
    find_frame_when_off_plate, get_n_seconds_before_takeoff
)
from jumpmetrics.signal_processing.filters import butterworth_filter

class WarningCatcher(logging.Handler):
    """Class for catching warnings and saving them"""
    def __init__(self):
        super().__init__()
        self.warnings = []

    def emit(self, record):
        if record.levelno == logging.WARNING:
            self.warnings.append(record.getMessage())

    def reset(self):
        """Reset warnings"""
        self.warnings = []

# Set up the warning catcher
warning_catcher = WarningCatcher()
logging.getLogger().addHandler(warning_catcher)

CREATE_PLOTS = False
main_dir = os.path.join(os.getcwd(), 'analyses', 'study_2')
data_dir = os.path.join(main_dir, 'raw_data')
fig_dir = os.path.join(main_dir, 'figures')
kinematic_data_dir = os.path.join(main_dir, 'kinematic_data')
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
    pid_data_dir = os.path.join(kinematic_data_dir, pid, trial_num)
    if not os.path.exists(pid_fig_dir):
        os.makedirs(pid_fig_dir)
    if not os.path.exists(pid_data_dir):
        os.makedirs(pid_data_dir)
    participant_filters = xlsx_data_condensed[xlsx_data_condensed['pid'] == pid]
    cutoff_frequency = participant_filters['group_cutoff'].item()
    try:
        tmp_df = pd.DataFrame()
        tmp_df['file_prefix'] = [FILE_PREFIX]
        tmp_df['cutoff_type'] = ['group_cutoff']
        cutoff_frequency = participant_filters['group_cutoff'].item()
        pid_fig_dir_cutoff_type = os.path.join(pid_fig_dir, 'group_cutoff')
        if not os.path.exists(pid_fig_dir_cutoff_type):
            os.makedirs(pid_fig_dir_cutoff_type)
        tmp_df['cutoff_frequency'] = [cutoff_frequency]
        tmp_force_df = load_raw_force_data_with_no_column_headers(filepath)
        full_summed_force = sum_dual_force_components(tmp_force_df)
        frame = find_first_frame_where_force_exceeds_threshold(
            force_trace=full_summed_force,
            threshold=1000
        )
        takeoff_frame = find_frame_when_off_plate(
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
        if CREATE_PLOTS:
            # Create Waveform Plots
            CMJ.plot_waveform(
                waveform_type='force',
                title=filename + '_group_cutoff',
                savefig=True,
                figname=os.path.join(pid_fig_dir_cutoff_type, 'force.png')
            )
            CMJ.plot_waveform(
                waveform_type='acceleration',
                title=filename + '_group_cutoff',
                savefig=True,
                figname=os.path.join(pid_fig_dir_cutoff_type, 'acceleration.png')
            )
            CMJ.plot_waveform(
                waveform_type='velocity',
                title=filename + '_group_cutoff',
                savefig=True,
                figname=os.path.join(pid_fig_dir_cutoff_type, 'velocity.png')
            )
            CMJ.plot_waveform(
                waveform_type='displacement',
                title=filename + '_group_cutoff',
                savefig=True,
                figname=os.path.join(pid_fig_dir_cutoff_type, 'displacement.png')
            )
            # Bespoke Plots and Dataframes
            # Force-Velocity
            plt.plot(
                CMJ.velocity_series[CMJ.start_of_unweighting_phase:],
                CMJ.force_series.iloc[CMJ.start_of_unweighting_phase:],
                color='blue'
            )
            plt.xlabel('Velocity (m/s)')
            plt.ylabel('Force (N)')
            plt.title(pid)
            # Highlight the start of the trial
            plt.plot(
                CMJ.velocity_series[CMJ.start_of_unweighting_phase],
                CMJ.force_series.iloc[CMJ.start_of_unweighting_phase], 'go',
                label='Start of Unweighting Phase'
            )  # green circle
            # Highlight the end of the trial
            plt.plot(
                CMJ.velocity_series[-1],
                CMJ.force_series.iloc[-1], 'ro',
                label='Takeoff'
            ) # red circle
            plt.legend()
            plt.savefig(os.path.join(pid_fig_dir_cutoff_type, 'force_velocity.png'), dpi=300)
            plt.clf()
            # Force-Displacement
            plt.plot(
                CMJ.displacement_series[CMJ.start_of_unweighting_phase:],
                CMJ.force_series.iloc[CMJ.start_of_unweighting_phase:],
                color='blue'
            )
            plt.xlabel('Displacement (m)')
            plt.ylabel('Force (N)')
            plt.title(pid)
            # Highlight the start of the trial
            plt.plot(
                CMJ.displacement_series[CMJ.start_of_unweighting_phase],
                CMJ.force_series.iloc[CMJ.start_of_unweighting_phase], 'go',
                label='Start of Unweighting Phase'
            )  # green circle
            # Highlight the end of the trial
            plt.plot(
                CMJ.displacement_series[-1],
                CMJ.force_series.iloc[-1], 'ro',
                label='Takeoff'
            ) # red circle
            plt.legend()
            plt.savefig(os.path.join(pid_fig_dir_cutoff_type, 'force_displacement.png'), dpi=300)
            plt.clf()
            # Velocity-Displacement
            plt.plot(
                CMJ.displacement_series[CMJ.start_of_unweighting_phase:],
                CMJ.velocity_series[CMJ.start_of_unweighting_phase:],
                color='blue'
            )
            plt.xlabel('Displacement (m)')
            plt.ylabel('Velocity (m/s)')
            plt.title(pid)
            # Highlight the start of the trial
            plt.plot(
                CMJ.displacement_series[CMJ.start_of_unweighting_phase],
                CMJ.velocity_series[CMJ.start_of_unweighting_phase], 'go',
                label='Start of Unweighting Phase'
            )  # green circle
            # Highlight the end of the trial
            plt.plot(
                CMJ.displacement_series[-1],
                CMJ.velocity_series[-1], 'ro',
                label='Takeoff'
            ) # red circle
            plt.legend()
            plt.savefig(os.path.join(pid_fig_dir_cutoff_type, 'velocity_displacement.png'), dpi=300)
            plt.clf()

        CMJ.save_kinematic_dataframe(
            dataframe_filepath=os.path.join(pid_data_dir, 'kinematic_data.csv')
        )
        force_series_df = CMJ.force_series.reset_index()
        force_series_df.columns = ['frame_number', 'force']  # Rename the columns
        # Save the DataFrame to a .csv file with specified column names
        force_series_df.to_csv(os.path.join(pid_data_dir, 'force_series.csv'), index=False)
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
