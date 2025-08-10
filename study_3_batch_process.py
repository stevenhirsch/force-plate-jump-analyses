"""Batch processing script for study 3"""
import os
import logging
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from jumpmetrics.core.processors import (
    ForceTimeCurveCMJTakeoffProcessor, ForceTimeCurveSQJTakeoffProcessor
)
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

# Parsing funcs
def parse_jump_info(jump_info):
    """Function to parse jump info"""
    jump_type = jump_info[:3]
    file_code = jump_info[3:].split('.txt')[0]
    overall_recording_num = file_code[-2:]
    rest_of_file_code = file_code[:-2]
    num_zeros = rest_of_file_code.count('0')
    trial_num_in_block = rest_of_file_code[0] if num_zeros == 3 else rest_of_file_code[0:2]
    return jump_type, overall_recording_num, trial_num_in_block


def parse_filename_metadata(fname):
    """Function to parse filename metadata"""
    sport, pid, jump_info = fname.split('_')
    jump_type, overall_recording_num, trial_num_in_block = parse_jump_info(
        jump_info=jump_info
    )
    filename_metadata = {
        'sport': sport,
        'pid': pid,
        'jump_type': jump_type,
        'trial_num_in_block': int(trial_num_in_block),
        'overall_recording_num': int(overall_recording_num)

    }
    return filename_metadata

CREATE_PLOTS = False
main_dir = os.path.join(os.getcwd(), 'analyses', 'study_3')
data_dir = os.path.join(main_dir, 'raw_data')
fig_dir = os.path.join(main_dir, 'figures')
kinematic_data_dir = os.path.join(main_dir, 'kinematic_data')
all_filenames = os.listdir(data_dir)
# xlsx_data_filepath = os.path.join(main_dir, 'signal_conditioning_filter_cutoffs.xlsx')
# xlsx_data = pd.read_excel(xlsx_data_filepath)
full_cmj_dataset = pd.DataFrame()
full_sqj_dataset = pd.DataFrame()
# xlsx_data['pid'] = xlsx_data['file_prefix'].str.split('_').str[0]
# xlsx_data_condensed = xlsx_data.groupby('pid').mean(numeric_only=True)
# xlsx_data_condensed.drop(['trial_cutoff', 'participant_cutoff', 'literature_cutoff'], axis=1, inplace=True)
# xlsx_data_condensed.reset_index(inplace=True)
corrupt_filenames = [
    'FHOC_P27_CMJ100002.txt',
    'FHOC_P12_SQT200006.txt'
]

for filename in tqdm(all_filenames):
    if filename in corrupt_filenames:
        continue
    warning_catcher.reset()
    filepath = os.path.join(data_dir, filename)
    if '.DS_Store' in filepath:
        continue
    file_metadata = parse_filename_metadata(filename)
    pid = file_metadata['pid']
    sport=file_metadata['sport']  # not needed right now, but will parse anyways
    jump_type = file_metadata['jump_type']
    trial_num_in_block = file_metadata['trial_num_in_block']
    overall_recording_number = file_metadata['overall_recording_num']
    TRIAL_NUM = str(trial_num_in_block) + '_' + str(overall_recording_number)
    if jump_type in ['SQT', 'CMJ']:
        pid_fig_dir = os.path.join(fig_dir, jump_type, pid, TRIAL_NUM)
        pid_data_dir = os.path.join(kinematic_data_dir, jump_type, pid, TRIAL_NUM)
    else:
        raise ValueError(
                f'jump_type must be "SQT" or "CMJ", but was {jump_type}'
        )
    if not os.path.exists(pid_fig_dir):
        os.makedirs(pid_fig_dir)
    if not os.path.exists(pid_data_dir):
        os.makedirs(pid_data_dir)
    CUTOFF_FREQUENCY = 50  # placeholder for now
    try:
        tmp_df = pd.DataFrame()
        tmp_df['sport'] = [sport]
        tmp_df['cutoff_type'] = ['literature_cutoff']
        tmp_df['jump_type'] = [jump_type]
        tmp_df['trial_num_in_block'] = [trial_num_in_block]
        tmp_df['overall_recording_number'] = [overall_recording_number]
        pid_fig_dir_cutoff_type = os.path.join(pid_fig_dir, 'literature_cutoff')
        if not os.path.exists(pid_fig_dir_cutoff_type):
            os.makedirs(pid_fig_dir_cutoff_type)
        tmp_df['cutoff_frequency'] = [CUTOFF_FREQUENCY]
        SAMPLING_FREQUENCY = 2000 if file_metadata['pid'] in ['P09'] else 1000
        tmp_force_df = load_raw_force_data_with_no_column_headers(filepath)
        full_summed_force = sum_dual_force_components(tmp_force_df)
        THRESHOLD = 750 if file_metadata['pid'] in ['P08'] else 1000
        frame = find_first_frame_where_force_exceeds_threshold(
            force_trace=full_summed_force,
            threshold=THRESHOLD
        )
        takeoff_frame = find_frame_when_off_plate(
            force_trace=full_summed_force.iloc[frame:],
            sampling_frequency=SAMPLING_FREQUENCY
        )
        TIME_BEFORE_TAKEOFF = 2
        cropped_force_trace = get_n_seconds_before_takeoff(
            force_trace=full_summed_force,
            sampling_frequency=SAMPLING_FREQUENCY,
            takeoff_frame=takeoff_frame,
            n=TIME_BEFORE_TAKEOFF
        )
        filtered_force_series = butterworth_filter(
            arr=cropped_force_trace,
            cutoff_frequency=CUTOFF_FREQUENCY,
            fps=SAMPLING_FREQUENCY,
            padding=SAMPLING_FREQUENCY
        )
        if jump_type == 'SQT':
            jump = ForceTimeCurveSQJTakeoffProcessor(
                force_series=filtered_force_series,
                sampling_frequency=SAMPLING_FREQUENCY
            )
        elif jump_type == 'CMJ':
            jump = ForceTimeCurveCMJTakeoffProcessor(
                force_series=filtered_force_series,
                sampling_frequency=SAMPLING_FREQUENCY
            )
        else:
            raise ValueError(
                f'jump_type must be "SQT" or "CMJ", but was {jump_type}'
        )
        jump.get_jump_events()
        jump.compute_jump_metrics()
        jump.create_jump_metrics_dataframe(pid=pid)
        jump.create_kinematic_dataframe()
        # Create Waveform Plots
        if CREATE_PLOTS:
            jump.plot_waveform(
                waveform_type='force',
                title=filename + '_literature_cutoff',
                savefig=True,
                figname=os.path.join(pid_fig_dir_cutoff_type, 'force.png')
            )
            jump.plot_waveform(
                waveform_type='acceleration',
                title=filename + '_literature_cutoff',
                savefig=True,
                figname=os.path.join(pid_fig_dir_cutoff_type, 'acceleration.png')
            )
            jump.plot_waveform(
                waveform_type='velocity',
                title=filename + '_literature_cutoff',
                savefig=True,
                figname=os.path.join(pid_fig_dir_cutoff_type, 'velocity.png')
            )
            jump.plot_waveform(
                waveform_type='displacement',
                title=filename + '_literature_cutoff',
                savefig=True,
                figname=os.path.join(pid_fig_dir_cutoff_type, 'displacement.png')
            )
            # Bespoke Plots and Dataframes
            # Force-Velocity
            if jump_type == 'CMJ':
                start_event = jump.start_of_unweighting_phase
            elif jump_type == 'SQT':
                start_event = jump.start_of_propulsive_phase
            else:
                raise ValueError(
                    f'jump_type must be "SQT" or "CMJ", but was {jump_type}'
            )  # unnecessary since other check will catch this, but have it here anyways
            plt.plot(
                jump.velocity_series[start_event:],
                jump.force_series.iloc[start_event:],
                color='blue'
            )
            plt.xlabel('Velocity (m/s)')
            plt.ylabel('Force (N)')
            plt.title(pid)
            # Highlight the start of the trial
            plt.plot(
                jump.velocity_series[start_event],
                jump.force_series.iloc[start_event], 'go',
                label='Start of Jump'
            )  # green circle
            # Highlight the end of the trial
            plt.plot(
                jump.velocity_series[-1],
                jump.force_series.iloc[-1], 'ro',
                label='Takeoff'
            ) # red circle
            plt.legend()
            plt.savefig(os.path.join(pid_fig_dir_cutoff_type, 'force_velocity.png'), dpi=300)
            plt.clf()
            # Force-Displacement
            plt.plot(
                jump.displacement_series[start_event:],
                jump.force_series.iloc[start_event:],
                color='blue'
            )
            plt.xlabel('Displacement (m)')
            plt.ylabel('Force (N)')
            plt.title(pid)
            # Highlight the start of the trial
            plt.plot(
                jump.displacement_series[start_event],
                jump.force_series.iloc[start_event], 'go',
                label='Start of Jump'
            )  # green circle
            # Highlight the end of the trial
            plt.plot(
                jump.displacement_series[-1],
                jump.force_series.iloc[-1], 'ro',
                label='Takeoff'
            ) # red circle
            plt.legend()
            plt.savefig(os.path.join(pid_fig_dir_cutoff_type, 'force_displacement.png'), dpi=300)
            plt.clf()
            # Velocity-Displacement
            plt.plot(
                jump.displacement_series[start_event:],
                jump.velocity_series[start_event:],
                color='blue'
            )
            plt.xlabel('Displacement (m)')
            plt.ylabel('Velocity (m/s)')
            plt.title(pid)
            # Highlight the start of the trial
            plt.plot(
                jump.displacement_series[start_event],
                jump.velocity_series[start_event], 'go',
                label='Start of Jump'
            )  # green circle
            # Highlight the end of the trial
            plt.plot(
                jump.displacement_series[-1],
                jump.velocity_series[-1], 'ro',
                label='Takeoff'
            ) # red circle
            plt.legend()
            plt.savefig(os.path.join(pid_fig_dir_cutoff_type, 'velocity_displacement.png'), dpi=300)
            plt.clf()

        jump.save_kinematic_dataframe(
            dataframe_filepath=os.path.join(pid_data_dir, 'kinematic_data.csv')
        )
        force_series_df = jump.force_series.reset_index()
        force_series_df.columns = ['frame_number', 'force']  # Rename the columns
        # Save the DataFrame to a .csv file with specified column names
        force_series_df.to_csv(os.path.join(pid_data_dir, 'force_series.csv'), index=False)
        tmp_df = pd.concat([
            tmp_df, jump.jump_metrics_dataframe
        ], axis=1)
        if jump_type == 'SQT':
            full_sqj_dataset = pd.concat([
                full_sqj_dataset, tmp_df
            ], axis=0)
        elif jump_type == 'CMJ':
            full_cmj_dataset = pd.concat([
                full_cmj_dataset, tmp_df
            ], axis=0)
        else:
            raise ValueError(
                f'jump_type must be "SQT" or "CMJ", but was {jump_type}'
            )  # again, redundant
    except ValueError as e:
        print(f'Skipping {filename} due to ValueError')
        print('')
        print(f"Specific ValueErrror: {e}")
        jump.plot_waveform(waveform_type='force', title=filename + '_literature_cutoff')
    if warning_catcher.warnings:
        print(f"Warnings occurred while processing {filename}:")
        for warning in warning_catcher.warnings:
            print(f"  - {warning}")

cmj_processed_file_name = os.path.join(main_dir, 'batch_processed_cmj_data.csv')
sqj_processed_file_name = os.path.join(main_dir, 'batch_processed_sqj_data.csv')
print('')
sort_order = [
    'PID', 'overall_recording_number', 'trial_num_in_block'
]
print('Saving results...')
full_sqj_dataset = full_sqj_dataset.sort_values(by=sort_order)
full_cmj_dataset = full_cmj_dataset.sort_values(by=sort_order)
full_sqj_dataset.to_csv(sqj_processed_file_name, index=False)
full_cmj_dataset.to_csv(cmj_processed_file_name, index=False)
