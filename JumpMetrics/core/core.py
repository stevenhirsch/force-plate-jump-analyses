import numpy as np  # library for dealing with arrays of numbers
import pandas as pd  # library for loading data and creating dataframes
import logging  # for logging errors
import matplotlib.pyplot as plt  # for plotting data
from scipy import signal
from JumpMetrics.signal_processing.numerical import (
    compute_derivative, compute_integral_of_signal, integrate_area
)
from JumpMetrics.events.cmj_events import (
    find_unweighting_start, get_start_of_propulsive_phase_using_displacement,
    get_start_of_braking_phase_using_velocity, get_peak_force_event
)
from JumpMetrics.metrics.cmj_metrics import (
    compute_rfd, get_bodyweight, compute_average_force_between_events,
    compute_jump_height_from_net_vertical_impulse, compute_jump_height_from_takeoff_velocity
)


class ForceTimeCurveCMJTakeoffProcessor:
    def __init__(self, force_series, sampling_frequency=1000):
        self.force_series = force_series
        if type(self.force_series) != pd.Series:
            self.force_series = pd.Series(self.force_series)
        self.body_weight = get_bodyweight(force_series = self.force_series)
        self.body_mass_kg = self.body_weight / 9.81
        # relevant kinematic series
        self.acceleration_series = (self.force_series - self.body_weight) / self.body_mass_kg
        self.velocity_series = compute_integral_of_signal(self.acceleration_series, sampling_frequency=sampling_frequency)
        self.displacement_series = compute_integral_of_signal(self.velocity_series, sampling_frequency=sampling_frequency)
        self.time = pd.Series((self.force_series.index + 1) / sampling_frequency)
        # extra variables to be used in other functions
        self.sampling_frequency = sampling_frequency
        self.force_series_minus_bodyweight = self.force_series - self.body_weight
        # defining empty dataframes
        self.jump_metrics = {}
        self.jump_metrics_dataframe = pd.DataFrame()
        self.kinematic_waveform_dataframe = pd.DataFrame()
        # events
        self.start_of_unweighting_phase = None
        self.start_of_braking_phase = None
        self.start_of_propulsive_phase = None
        self.peak_force_frame = None

    def get_jump_events(self, velocity_threshold_to_define_start_of_jump=0.03):
        self.start_of_unweighting_phase = find_unweighting_start(
            force_data=self.force_series,
            sample_rate=self.sampling_frequency,
            quiet_period=0.5,  # 0.5seconds
            duration_check=0.175  # 175 milliseconds of unweighting to avoid false positives
        )
        self.start_of_propulsive_phase = get_start_of_propulsive_phase_using_displacement(
            self.displacement_series
        )
        self.start_of_braking_phase = get_start_of_braking_phase_using_velocity(
            self.velocity_series
        )
        self.peak_force_frame = get_peak_force_event(
            force_series=self.force_series,
            start_of_propulsive_phase=self.start_of_propulsive_phase
        )

    def compute_jump_metrics(self):
        events_list = [self.start_of_unweighting_phase, self.start_of_braking_phase, self.start_of_propulsive_phase, self.peak_force_frame]
        if any(value is None for value in events_list):
            raise ValueError(f"The following events need to be defined first: {events_list}")
        else:
            self.jump_metrics['propulsive_peakforce_rfd_slope_between_events'] = compute_rfd(
                force_trace = self.force_series,
                window_start = self.start_of_propulsive_phase,
                window_end = self.peak_force_frame,
                sampling_frequency=self.sampling_frequency,
                method='average'
            )
            self.jump_metrics['propulsive_peakforce_rfd_instantaneous_average_between_events'] = compute_rfd(
                force_trace = self.force_series,
                window_start = self.start_of_propulsive_phase,
                window_end = self.peak_force_frame,
                sampling_frequency=self.sampling_frequency,
                method='instantaneous'
            )
            self.jump_metrics['propulsive_peakforce_rfd_instantaneous_peak_between_events'] = compute_rfd(
                force_trace = self.force_series,
                window_start = self.start_of_propulsive_phase,
                window_end = self.peak_force_frame,
                sampling_frequency=self.sampling_frequency,
                method='peak'
            )
            self.jump_metrics['braking_peakforce_rfd_slope_between_events'] = compute_rfd(
                force_trace = self.force_series,
                window_start = self.start_of_braking_phase,
                window_end = self.peak_force_frame,
                sampling_frequency=self.sampling_frequency,
                method='average'
            )
            self.jump_metrics['braking_peakforce_rfd_instantaneous_average_between_events'] = compute_rfd(
                force_trace = self.force_series,
                window_start = self.start_of_braking_phase,
                window_end = self.peak_force_frame,
                sampling_frequency=self.sampling_frequency,
                method='instantaneous'
            )
            self.jump_metrics['braking_peakforce_rfd_instantaneous_peak_between_events'] = compute_rfd(
                force_trace = self.force_series,
                window_start = self.start_of_braking_phase,
                window_end = self.peak_force_frame,
                sampling_frequency=self.sampling_frequency,
                method='peak'
            )
            self.jump_metrics['braking_propulsive_rfd_slope_between_events'] = compute_rfd(
                force_trace = self.force_series,
                window_start = self.start_of_braking_phase,
                window_end = self.start_of_propulsive_phase,
                sampling_frequency=self.sampling_frequency,
                method='average'
            )
            self.jump_metrics['braking_propulsive_rfd_instantaneous_average_between_events'] = compute_rfd(
                force_trace = self.force_series,
                window_start = self.start_of_braking_phase,
                window_end = self.start_of_propulsive_phase,
                sampling_frequency=self.sampling_frequency,
                method='instantaneous'
            )
            self.jump_metrics['braking_propulsive_rfd_instantaneous_peak_between_events'] = compute_rfd(
                force_trace = self.force_series,
                window_start = self.start_of_braking_phase,
                window_end = self.start_of_propulsive_phase,
                sampling_frequency=self.sampling_frequency,
                method='peak'
            )
            if self.peak_force_frame > self.start_of_braking_phase:
                self.jump_metrics['braking_net_vertical_impulse'] = integrate_area(
                    time=self.time.iloc[self.start_of_braking_phase:self.peak_force_frame],
                    signal=self.force_series_minus_bodyweight.iloc[self.start_of_braking_phase:self.peak_force_frame]
                )
            else:
                logging.warning(f"Peak force occurred before or at the start of the braking phase, so impulse between events is invalid")
                self.jump_metrics['braking_net_vertical_impulse'] = np.nan
            if self.peak_force_frame > self.start_of_propulsive_phase:
                self.jump_metrics['propulsive_net_vertical_impulse'] = integrate_area(
                    time=self.time.iloc[self.start_of_propulsive_phase:self.peak_force_frame],
                    signal=self.force_series_minus_bodyweight.iloc[self.start_of_propulsive_phase:self.peak_force_frame]
                )
            else:
                logging.warning(f"Peak force occurred before or at the start of the propulsive phase, so impulse between events is invalid")
                self.jump_metrics['propulsive_net_vertical_impulse'] = np.nan
            if self.start_of_propulsive_phase > self.start_of_braking_phase:
                self.jump_metrics['braking_to_propulsive_net_vertical_impulse'] = integrate_area(
                    time=self.time.iloc[self.start_of_braking_phase:self.start_of_propulsive_phase],
                    signal=self.force_series_minus_bodyweight.iloc[self.start_of_braking_phase:self.start_of_propulsive_phase]
                )
            else:
                logging.warning(f"Braking phase occured before or at the start of the propulsive phase, so impulse between events is invalid")
                self.jump_metrics['braking_to_propulsive_net_vertical_impulse'] = np.nan

            self.jump_metrics['total_net_vertical_impulse'] = integrate_area(
                time=self.time,
                signal=self.force_series_minus_bodyweight
            )
            self.jump_metrics['peak_force'] = self.force_series.iloc[self.peak_force_frame]
            self.jump_metrics['average_force_of_braking_phase'] = compute_average_force_between_events(
                force_trace=self.force_series,
                window_start=self.start_of_braking_phase,
                window_end=self.start_of_propulsive_phase
            )
            self.jump_metrics['average_force_of_propulsive_phase'] = compute_average_force_between_events(
                force_trace=self.force_series,
                window_start=self.start_of_propulsive_phase,
                window_end=-1  # take the last frame
            )
            self.jump_metrics['takeoff_velocity'] = self.velocity_series[-1]
            self.jump_metrics['jump_height_takeoff_velocity'] = compute_jump_height_from_takeoff_velocity(
                self.jump_metrics['takeoff_velocity'].item()
            )
            self.jump_metrics['jump_height_net_vertical_impulse'] = compute_jump_height_from_net_vertical_impulse(
                net_vertical_impulse=self.jump_metrics['total_net_vertical_impulse'].item(),
                body_mass_kg=self.body_mass_kg
            )
            self.jump_metrics['movement_time'] = (self.peak_force_frame - self.start_of_unweighting_phase) / self.sampling_frequency
            self.jump_metrics['unweighting_time'] = (self.start_of_braking_phase - self.start_of_unweighting_phase) / self.sampling_frequency
            self.jump_metrics['braking_time'] = (self.start_of_propulsive_phase - self.start_of_braking_phase) / self.sampling_frequency
            self.jump_metrics['propulsive_time'] = (self.peak_force_frame - self.start_of_propulsive_phase) / self.sampling_frequency
            self.jump_metrics['lowering_displacement'] = np.min(self.displacement_series)
            self.jump_metrics['frame_start_of_unweighting_phase'] = self.start_of_unweighting_phase
            self.jump_metrics['frame_start_of_breaking_phase'] = self.start_of_braking_phase
            self.jump_metrics['frame_start_of_propulsive_phase'] = self.start_of_propulsive_phase
            self.jump_metrics['frame_peak_force'] = self.peak_force_frame

    def create_jump_metrics_dataframe(self, pid):
        if not self.jump_metrics:
            raise ValueError("Must run `compute_jump_metrics()` before creating a dataframe")
        else:
            self.jump_metrics_dataframe['PID'] = [pid]
            self.jump_metrics_dataframe = pd.concat([self.jump_metrics_dataframe, pd.DataFrame(self.jump_metrics, index=[0])], axis=1)

    def plot_waveform(self, waveform_type, title=None, savefig=False, figname=None):
        # valid_waveform_types = ['force', 'acceleration', 'velocity', 'displacement']
        waveform_dict = {
            # waveform_type: (waveform, ylabel)
            'force': (self.force_series, 'Ground Reaction Force (N)'),
            'acceleration': (self.acceleration_series, 'Acceleration (m/s^2)'),
            'velocity': (self.velocity_series, 'Velocity (m/s)'),
            'displacement': (self.displacement_series, 'Displacement (m)')
        }
        if waveform_type not in waveform_dict.keys():
            raise ValueError(f"{waveform_type} is invalid. Please specify one of: {waveform_dict.keys()}.")
        else:
            waveform, ylabel = waveform_dict[waveform_type]
            plt.plot(self.time, waveform)
            if waveform_type == 'force':
                plt.axhline(self.body_weight, color='red', linestyle='--', label=f'Bodyweight ({round(self.body_weight, 2)}N)')
            if self.start_of_unweighting_phase is not None:
                plt.axvline(self.start_of_unweighting_phase / self.sampling_frequency, color='green', label='Start of Unweighting Phase')
            if self.start_of_braking_phase is not None:
                plt.axvline(self.start_of_braking_phase / self.sampling_frequency, color='yellow', label='Start of Braking Phase')
            if self.start_of_propulsive_phase is not None:
                plt.axvline(self.start_of_propulsive_phase / self.sampling_frequency, color='orange', label='Start of Propulsive Phase')
            if self.peak_force_frame is not None:
                plt.axvline(self.peak_force_frame / self.sampling_frequency, color='blue', linestyle='--', label='Peak Force')
                plt.legend()
            if title is not None and type(title) == str:
                plt.title(title)
            plt.ylabel(ylabel)
            plt.xlabel('Time (sec)')
            if type(savefig) != bool:
                raise ValueError('savefig must be set to true or false (i.e., a boolean)')
            elif savefig == False:
                plt.show()
            elif savefig == True and (figname is None or type(figname) != str):
                raise ValueError('If specifying savefig, please provide a valid filename as a string')
            else:
                plt.savefig(figname, dpi=300)
                plt.close()

    def save_jump_metrics_dataframe(self, dataframe_filepath):
        if len(self.jump_metrics_dataframe) == 0:
            raise ValueError("Must run `create_dataframe` before trying to save a dataframe")
        else:
            self.jump_metrics_dataframe.to_csv(dataframe_filepath, index=False)

    def create_kinematic_dataframe(self):
        self.kinematic_waveform_dataframe = pd.DataFrame({
            # 'pid': [pid] * len(self.force_series),
            'acceleration': self.acceleration_series,
            'velocity': self.velocity_series,
            'displacement': self.displacement_series
        })

    def save_kinematic_dataframe(self, dataframe_filepath):
        if len(self.kinematic_waveform_dataframe) == 0:
            raise ValueError("Must run `create_dataframe` before trying to save a dataframe")
        else:
            self.kinematic_waveform_dataframe.to_csv(dataframe_filepath, index=False)
