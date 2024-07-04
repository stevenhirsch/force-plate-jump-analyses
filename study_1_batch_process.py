"""Batch processing script for study 1"""
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from jumpmetrics.core.core import ForceTimeCurveCMJTakeoffProcessor
from jumpmetrics.core.io import load_cropped_force_data
from jumpmetrics.signal_processing.filters import butterworth_filter


main_dir = os.path.join(os.getcwd(), 'analyses', 'study_1')
data_dir = os.path.join(main_dir, 'raw_data')
fig_dir = os.path.join(main_dir, 'figures')
kinematic_data_dir = os.path.join(main_dir, 'kinematic_data')
all_filenames = os.listdir(data_dir)
xlsx_data_filepath = os.path.join(main_dir, 'signal_conditioning_filter_cutoffs.xlsx')
xlsx_data = pd.read_excel(xlsx_data_filepath)
full_dataset = pd.DataFrame()

for filename in tqdm(all_filenames):
    filepath = os.path.join(data_dir, filename)
    FILE_PREFIX = '_'.join(filename.split('_')[0:2])
    pid = FILE_PREFIX.split('_', maxsplit=1)[0]
    trial_num = FILE_PREFIX.split('_')[1]
    pid_fig_dir = os.path.join(fig_dir, pid, trial_num)
    pid_data_dir = os.path.join(kinematic_data_dir, pid, trial_num)
    if not os.path.exists(pid_fig_dir):
        os.makedirs(pid_fig_dir)
    if not os.path.exists(pid_data_dir):
        os.makedirs(pid_data_dir)
    participant_filters = xlsx_data[xlsx_data['file_prefix'] == FILE_PREFIX]
    for col in participant_filters.columns:
        if col == 'file_prefix':
            continue
        else:
            try:
                tmp_df = pd.DataFrame()
                tmp_df['file_prefix'] = [FILE_PREFIX]
                tmp_df['cutoff_type'] = [col]
                cutoff_frequency = participant_filters[col].item()
                pid_fig_dir_cutoff_type = os.path.join(pid_fig_dir, col)
                if not os.path.exists(pid_fig_dir_cutoff_type):
                    os.makedirs(pid_fig_dir_cutoff_type)
                tmp_df['cutoff_frequency'] = [cutoff_frequency]
                force_series = load_cropped_force_data(
                    filepath=filepath,
                    freq=None
                )
                filtered_force_series = butterworth_filter(
                    arr=force_series,
                    cutoff_frequency=cutoff_frequency,
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
                CMJ.create_jump_metrics_dataframe(pid=pid)
                CMJ.create_kinematic_dataframe()
                CMJ.plot_waveform(
                    waveform_type='force',
                    title=filename + '_' + col,
                    savefig=True,
                    figname=os.path.join(pid_fig_dir_cutoff_type, 'force.png')
                )
                CMJ.plot_waveform(
                    waveform_type='acceleration',
                    title=filename + '_' + col,
                    savefig=True,
                    figname=os.path.join(pid_fig_dir_cutoff_type, 'acceleration.png')
                )
                CMJ.plot_waveform(
                    waveform_type='velocity',
                    title=filename + '_' + col,
                    savefig=True,
                    figname=os.path.join(pid_fig_dir_cutoff_type, 'velocity.png')
                )
                CMJ.plot_waveform(
                    waveform_type='displacement',
                    title=filename + '_' + col,
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
                    dataframe_filepath=os.path.join(pid_data_dir, col + '.csv')
                )
                # CMJ.force_series.to_csv(os.path.join(pid_data_dir, col + '_force_series.csv'))
                force_series_df = CMJ.force_series.reset_index()
                force_series_df.columns = ['frame_number', 'force']  # Rename the columns
                # Save the DataFrame to a .csv file with specified column names
                force_series_df.to_csv(os.path.join(pid_data_dir, col + '_force_series.csv'), index=False)
                tmp_df = pd.concat([
                    tmp_df, CMJ.jump_metrics_dataframe
                ], axis=1)
                full_dataset = pd.concat([
                    full_dataset, tmp_df
                ], axis=0)
            except ValueError as e:
                print(f'Skipping {filename}, {col} due to ValueError')
                print('')
                print(f"Specific ValueErrror: {e}")
                CMJ.plot_waveform(waveform_type='force', title=filename + '_' + col)

processed_file_name = os.path.join(main_dir, 'batch_processed_data.csv')
print('')
print('Saving results...')
full_dataset = full_dataset.sort_values(by='file_prefix')
full_dataset.to_csv(processed_file_name, index=False)
