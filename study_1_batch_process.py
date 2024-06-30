import os
from JumpMetrics.core.core import ForceTimeCurveCMJTakeoffProcessor
from JumpMetrics.core.io import load_cropped_force_data
from JumpMetrics.signal_processing.filters import butterworth_filter
import pandas as pd
from tqdm import tqdm


main_dir = os.path.join(os.getcwd(), 'analyses', 'study_1')
data_dir = os.path.join(main_dir, 'raw_data')
fig_dir = os.path.join(main_dir, 'figures')
all_filenames = os.listdir(data_dir)
xlsx_data_filepath = os.path.join(main_dir, 'signal_conditioning_filter_cutoffs.xlsx')
xlsx_data = pd.read_excel(xlsx_data_filepath)
full_dataset = pd.DataFrame()

for filename in tqdm(all_filenames):
    filepath = os.path.join(data_dir, filename)
    file_prefix = '_'.join(filename.split('_')[0:2])
    pid = file_prefix.split('_')[0]
    trial_num = file_prefix.split('_')[1]
    pid_fig_dir = os.path.join(fig_dir, pid, trial_num)
    if not os.path.exists(pid_fig_dir):
        os.makedirs(pid_fig_dir)
    participant_filters = xlsx_data[xlsx_data['file_prefix'] == file_prefix]
    for col in participant_filters.columns:
        if col == 'file_prefix':
            continue
        else:
            try:
                tmp_df = pd.DataFrame()
                tmp_df['file_prefix'] = [file_prefix]
                tmp_df['cutoff_type'] = [col]
                cutoff_frequency = participant_filters[col].item()
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
                CMJ.get_jump_events(
                    velocity_threshold_to_define_start_of_jump=0.03
                )
                CMJ.compute_jump_metrics()
                CMJ.create_jump_metrics_dataframe(pid=pid)
                CMJ.create_kinematic_dataframe()
                CMJ.plot_waveform(
                    waveform_type='force',
                    title=filename + '_' + col,
                    savefig=True,
                    figname=os.path.join(pid_fig_dir, col + '.png')
                )
                tmp_df = pd.concat([
                    tmp_df, CMJ.jump_metrics_dataframe
                ], axis=1)
                full_dataset = pd.concat([
                    full_dataset, tmp_df
                ], axis=0)
            except ValueError:
                print(f'Skipping {filename}, {col} due to ValueError')
                CMJ.plot_waveform(waveform_type='force', title=filename + '_' + col)

processed_file_name = os.path.join(main_dir, 'batch_processed_data.csv')
print('')
print('Saving results...')
full_dataset = full_dataset.sort_values(by='file_prefix')
full_dataset.to_csv(processed_file_name, index=False)