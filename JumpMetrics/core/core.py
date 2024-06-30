import numpy as np  # library for dealing with arrays of numbers
import pandas as pd  # library for loading data and creating dataframes
from scipy.signal import find_peaks  # Handy function for finding peaks from the SciPy library
import logging  # for logging errors
import matplotlib.pyplot as plt  # for plotting data
from scipy import signal
from scipy.signal import savgol_filter
from JumpMetrics.signal_processing.numerical import (
    compute_derivative, compute_integral_of_signal, integrate_area
)


# get bodyweight (assumes first n frames are "static" and represent someone's bodyweight in Newtons)
def get_bodyweight(force_series, n=500):
    average_of_first_n_frames = force_series.iloc[0:n].mean()
    return average_of_first_n_frames

def get_start_of_jump(series, threshold=0.05):
    frame = np.where(series < -threshold)[0][0]
    return frame

def get_start_of_jump_based_on_stance_sd(series, n=500):
    average_weight = series.iloc[0:n].mean()
    sd_weight = series.iloc[0:n].std()
    series_average_diff = series - average_weight
    frame = np.where(series_average_diff < -sd_weight*5)[0][0]  # 5 standard deviations below the SD during stance
    return frame

def get_start_of_jump_based_on_accel(accel_series, threshold=-1):
    jerk = accel_series.diff()
    frame = np.where(jerk <= threshold)[0][0]
    return frame

def find_unweighting_start(force_data, sample_rate, quiet_period=1, threshold_factor=5, window_size=0.2, duration_check=0.1):
    # Convert window size from seconds to samples
    window_samples = int(window_size * sample_rate)
    
    # Smooth the force data using Savitzky-Golay filter
    smoothed_force = savgol_filter(force_data, window_samples, 3)
    
    # Calculate mean and standard deviation of the quiet period
    quiet_samples = int(quiet_period * sample_rate)
    quiet_force = force_data[:quiet_samples]
    mean_force = np.mean(quiet_force)
    std_force = np.std(quiet_force)
    
    # Calculate the threshold
    threshold = mean_force - threshold_factor * std_force
    
    # Convert duration_check from seconds to samples
    duration_samples = int(duration_check * sample_rate)
    
    # Find the first point where smoothed force drops below threshold
    unweighting_start = None
    for i in range(quiet_samples, len(smoothed_force) - duration_samples):
        if smoothed_force[i] < threshold:
            # Check if force stays below threshold for the specified duration
            if np.all(smoothed_force[i:i+duration_samples] < threshold):
                unweighting_start = i
                break
    
    return unweighting_start

def get_start_of_concentric_phase(velocity_series, frame_of_eccentric_start):
    # frame = np.where(velocity_series[frame_of_eccentric_start:] > 0)[0][0]
    min_velocity_index = np.argmin(velocity_series)
    frame = np.where(velocity_series[min_velocity_index:] > 0)[0][0]
    return frame + min_velocity_index

# def get_start_of_braking_phase(velocity_series, unweighting_phase, propulsive_phase):
#     frame = find_peaks(velocity_series[unweighting_phase:propulsive_phase] * -1)[0]
#     return frame[0] + unweighting_phase

def get_start_of_braking_phase_using_velocity(velocity_series):
    return np.argmin(velocity_series)

def get_start_of_propulsive_phase_using_displacement(displacement_series):
    return np.argmin(displacement_series)

def get_peak_force_event(force_series, start_of_propulsive_phase):
    peaks, _ = find_peaks(
        force_series.iloc[start_of_propulsive_phase:], prominence=50
    )
    if len(peaks) == 0:
        peak_force_frame = force_series.iloc[start_of_propulsive_phase:].idxmax()
    else:
        peak_force_frame = peaks[0] + start_of_propulsive_phase
    return peak_force_frame

# Metrics
def compute_rfd(force_trace, window_start, window_end, sampling_frequency, method='average'):
    if window_end <= window_start:
        logging.warning("window_end is before or equal to window_start, so RFD will be invalid. Returning np.nan")
        return np.nan
    valid_methods = ['average', 'instantaneous', 'peak']
    if method == 'average':
        force_at_start = force_trace[window_start]
        force_at_end = force_trace[window_end]
        num_frames_between_points = window_end - window_start
        time_between_points = num_frames_between_points / sampling_frequency
        rfd = (force_at_end - force_at_start) / time_between_points
    elif method == 'instantaneous' or method == 'peak':
        force_trace_derivative = compute_derivative(force_trace, t=1/sampling_frequency, d=1)
        rfd_between_events = force_trace_derivative[window_start:window_end]
        if method == 'peak':
            rfd = np.max(rfd_between_events)
        else:
            rfd = np.mean(rfd_between_events)
    else:
        logging.error(f"{method} is not a valid method. Please select one of: {valid_methods}")
        rfd = None

    return rfd

def compute_jump_height_from_takeoff_velocity(takeoff_velocity):
    jump_height = (takeoff_velocity ** 2) / (2 * 9.81)
    return jump_height

def compute_jump_height_from_velocity_series(velocity_series):
    takeoff_velocity = velocity_series[-1]  # take the last frame
    jump_height = compute_jump_height_from_takeoff_velocity(takeoff_velocity)
    return jump_height

def compute_jump_height_from_net_vertical_impulse(net_vertical_impulse, body_mass_kg):
    impulse_takeoff_velocity = net_vertical_impulse / body_mass_kg  # impulse = momentum = mass * velocity
    jump_height = compute_jump_height_from_takeoff_velocity(takeoff_velocity=impulse_takeoff_velocity)
    return jump_height

def compute_average_force_between_events(force_trace, window_start, window_end):
    if window_end <= window_start and window_end > 0:
        logging.warning(f"window_end occurred before or at the same time as window_start, so average force between events is invalid. Returning np.nan")
        return np.nan
    average_force = force_trace.iloc[window_start:window_end].mean()
    return average_force


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
        # self.start_of_unweighting_phase = get_start_of_jump(
        #     series=self.velocity_series,
        #     threshold=velocity_threshold_to_define_start_of_jump
        # )
        # self.start_of_unweighting_phase = get_start_of_jump_based_on_stance_sd(
        #     series=self.force_series,
        #     n=500
        # )
        self.start_of_unweighting_phase = find_unweighting_start(
            force_data=self.force_series,
            sample_rate=self.sampling_frequency,
            quiet_period=0.5,  # 0.5seconds
            duration_check=0.175  # 175 milliseconds of unweighting to avoid false positives
        )
        # self.start_of_propulsive_phase = get_start_of_concentric_phase(
        #     velocity_series=self.velocity_series,
        #     frame_of_eccentric_start=self.start_of_unweighting_phase
        # )
        self.start_of_propulsive_phase = get_start_of_propulsive_phase_using_displacement(
            self.displacement_series
        )
        # self.start_of_braking_phase = get_start_of_braking_phase(
        #     self.velocity_series,
        #     self.start_of_unweighting_phase,
        #     self.start_of_propulsive_phase
        # )
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
