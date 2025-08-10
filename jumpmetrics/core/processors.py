"""Core class for computing jump events and metrics"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jumpmetrics.signal_processing.numerical import (
    compute_integral_of_signal, integrate_area
)
from jumpmetrics.events.cmj_events import (
    find_unweighting_start, get_start_of_propulsive_phase_using_displacement,
    get_start_of_braking_phase_using_velocity, get_peak_force_event
)
from jumpmetrics.metrics.metrics import (
    compute_rfd, get_bodyweight, compute_average_force_between_events,
    compute_jump_height_from_net_vertical_impulse, compute_jump_height_from_takeoff_velocity
)
from jumpmetrics.events.sqj_events import (
    get_start_of_propulsive_phase, get_sqj_peak_force_event, find_potential_unweighting
)
from jumpmetrics.events.landing_events import (
    get_end_of_landing_phase
)

PF_B4_BRAKING = 'Peak force occurred before or at the start of the braking phase, metrics between events invalid'
PF_B4_PROP = 'Peak force occurred before or at the start of the propulsive phase, metrics between events invalid'
BRAKE_B4_PROP = 'Braking phase occured before or at the start of the propulsive phase, metrics between events invalid'


class ForceTimeCurveTakeoffProcessor:
    """Base class for computing jump events and metrics"""

    def __init__(self, force_series, sampling_frequency=2000, weighing_time=0.4):
        self.force_series = np.array(force_series)
        self.sampling_frequency = sampling_frequency
        self.body_weight = get_bodyweight(
            force_series=self.force_series,
            sampling_frequency=self.sampling_frequency,
            sec=weighing_time
        )
        self.body_mass_kg = self.body_weight / 9.81

        # Kinematic series
        self.acceleration_series = (self.force_series - self.body_weight) / self.body_mass_kg
        self.velocity_series = compute_integral_of_signal(
            self.acceleration_series,
            sampling_frequency=sampling_frequency
        )
        self.displacement_series = compute_integral_of_signal(
            self.velocity_series,
            sampling_frequency=sampling_frequency
        )
        self.time = np.arange(0, len(self.force_series) / self.sampling_frequency, 1/self.sampling_frequency)

        # Additional variables
        self.force_series_minus_bodyweight = self.force_series - self.body_weight

        # Metrics and dataframes
        self.jump_metrics = {}
        self.jump_metrics_dataframe = pd.DataFrame()
        self.kinematic_waveform_dataframe = pd.DataFrame()

        # Events (to be defined in child classes)
        self.start_of_propulsive_phase = None
        self.peak_force_frame = None

    def get_jump_events(self):
        """To be implemented by child classes"""
        raise NotImplementedError("This method should be implemented by child classes")

    def compute_jump_metrics(self):
        """To be implemented by child classes"""
        raise NotImplementedError("This method should be implemented by child classes")

    def create_jump_metrics_dataframe(self, pid: str):
        """Function to create jump metrics dataframe"""
        if not self.jump_metrics:
            raise ValueError("Must run `compute_jump_metrics()` before creating a dataframe")
        self.jump_metrics_dataframe['PID'] = [pid]
        self.jump_metrics_dataframe = pd.concat(
            [self.jump_metrics_dataframe, pd.DataFrame(self.jump_metrics, index=[0])],
            axis=1
        )

    def save_jump_metrics_dataframe(self, dataframe_filepath: str):
        """Function to save jump metrics dataframe"""
        if len(self.jump_metrics_dataframe) == 0:
            raise ValueError("Must run `create_dataframe` before trying to save a dataframe")
        self.jump_metrics_dataframe.to_csv(dataframe_filepath, index=False)

    def create_kinematic_dataframe(self):
        """function to create kinematic dataframe"""
        self.kinematic_waveform_dataframe = pd.DataFrame({
            'acceleration': self.acceleration_series,
            'velocity': self.velocity_series,
            'displacement': self.displacement_series
        })

    def save_kinematic_dataframe(self, dataframe_filepath: str):
        """Function to save the kinematic dataframe"""
        if len(self.kinematic_waveform_dataframe) == 0:
            raise ValueError("Must run `create_dataframe` before trying to save a dataframe")
        self.kinematic_waveform_dataframe.to_csv(dataframe_filepath, index=False)

    def plot_waveform(self, waveform_type: str, title=None, savefig: bool = False, figname=None):
        """Function to plot waveforms"""
        waveform_dict = {
            'force': (self.force_series, 'Ground Reaction Force (N)'),
            'acceleration': (self.acceleration_series, 'Acceleration (m/s^2)'),
            'velocity': (self.velocity_series, 'Velocity (m/s)'),
            'displacement': (self.displacement_series, 'Displacement (m)')
        }
        if waveform_type not in waveform_dict:
            raise ValueError(f"{waveform_type} is invalid. Please specify one of: {waveform_dict.keys()}.")

        waveform, ylabel = waveform_dict[waveform_type]
        plt.figure(figsize=(10, 6))
        plt.plot(self.time, waveform)

        if waveform_type == 'force':
            plt.axhline(self.body_weight, color='red', linestyle='--',
                        label=f'Bodyweight ({round(self.body_weight, 2)}N)')

        self._plot_specific_events()

        plt.legend()
        if title:
            plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('Time (sec)')

        if savefig and figname:
            plt.savefig(figname, dpi=300)
            plt.close()
        elif not savefig:
            plt.show()
        else:
            raise ValueError('If specifying savefig, please provide a valid filename as a string')

    def _plot_specific_events(self):
        # This method will be overridden in child classes
        pass


class ForceTimeCurveCMJTakeoffProcessor(ForceTimeCurveTakeoffProcessor):
    """Class for computing CMJ takeoff events and metrics"""
    def __init__(self, force_series, sampling_frequency=2000, weighing_time=0.25):
        super().__init__(force_series, sampling_frequency, weighing_time)
        self.start_of_unweighting_phase = None
        self.start_of_braking_phase = None

    def get_jump_events(self, unweighting_phase_quiet_period=0.5, unweighting_phase_duration_check=0.175):
        """Function to get jump events for a ForceTimeCurveCMJTakeoffProcessor Class
        """
        self.start_of_unweighting_phase = find_unweighting_start(
            force_data=self.force_series,
            sample_rate=self.sampling_frequency,
            quiet_period=unweighting_phase_quiet_period,  # 0.5seconds
            duration_check=unweighting_phase_duration_check  # 175 milliseconds of unweighting to avoid false positives
        )
        self.start_of_braking_phase = get_start_of_braking_phase_using_velocity(
            velocity_series=self.velocity_series,
            start_of_unweighting_phase=self.start_of_unweighting_phase
        )
        self.start_of_propulsive_phase = get_start_of_propulsive_phase_using_displacement(
            displacement_series=self.displacement_series,
            start_of_braking_phase=self.start_of_braking_phase
        )
        self.peak_force_frame = get_peak_force_event(
            force_series=self.force_series,
            start_of_propulsive_phase=self.start_of_propulsive_phase
        )

    def compute_jump_metrics(self):
        """Function to compute CMJ metrics

        Raises:
            ValueError: If any events are None, a ValueError is raised since metrics cannot
            be defined without events
        """
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
                time=self.time[self.start_of_braking_phase:self.peak_force_frame],
                signal=self.force_series_minus_bodyweight[self.start_of_braking_phase:self.peak_force_frame]
            )
        else:
            logging.warning(PF_B4_BRAKING)
            self.jump_metrics['braking_net_vertical_impulse'] = np.nan
        if self.peak_force_frame > self.start_of_propulsive_phase:
            self.jump_metrics['propulsive_net_vertical_impulse'] = integrate_area(
                time=self.time[self.start_of_propulsive_phase:self.peak_force_frame],
                signal=self.force_series_minus_bodyweight[self.start_of_propulsive_phase:self.peak_force_frame]
            )
        else:
            logging.warning(PF_B4_PROP)
            self.jump_metrics['propulsive_net_vertical_impulse'] = np.nan
        if self.start_of_propulsive_phase > self.start_of_braking_phase:
            self.jump_metrics['braking_to_propulsive_net_vertical_impulse'] = integrate_area(
                time=self.time[self.start_of_braking_phase:self.start_of_propulsive_phase],
                signal=self.force_series_minus_bodyweight[
                    self.start_of_braking_phase:self.start_of_propulsive_phase
                ]
            )
        else:
            logging.warning(BRAKE_B4_PROP)
            self.jump_metrics['braking_to_propulsive_net_vertical_impulse'] = np.nan

        self.jump_metrics['total_net_vertical_impulse'] = integrate_area(
            time=self.time,
            signal=self.force_series_minus_bodyweight
        )
        self.jump_metrics['peak_force'] = self.force_series[self.peak_force_frame]
        self.jump_metrics['maximum_force'] = np.nanmax(self.force_series)
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
        if self.start_of_unweighting_phase is not None:
            self.jump_metrics['movement_time'] = (
                len(self.force_series) - self.start_of_unweighting_phase
            ) / self.sampling_frequency
            self.jump_metrics['unweighting_time'] = (
                self.start_of_braking_phase - self.start_of_unweighting_phase
            ) / self.sampling_frequency
        else:
            self.jump_metrics['movement_time'] = np.nan
            self.jump_metrics['unweighting_time'] = np.nan
        if self.start_of_braking_phase is not None:
            self.jump_metrics['braking_time'] = (
                self.start_of_propulsive_phase - self.start_of_braking_phase
            ) / self.sampling_frequency
            self.jump_metrics['propulsive_time'] = (
                len(self.force_series) - self.start_of_propulsive_phase
            ) / self.sampling_frequency
        else:
            self.jump_metrics['braking_time'] = np.nan
            self.jump_metrics['propulsive_time'] = np.nan
        self.jump_metrics['lowering_displacement'] = np.min(self.displacement_series)
        self.jump_metrics['frame_start_of_unweighting_phase'] = self.start_of_unweighting_phase
        self.jump_metrics['frame_start_of_breaking_phase'] = self.start_of_braking_phase
        self.jump_metrics['frame_start_of_propulsive_phase'] = self.start_of_propulsive_phase
        self.jump_metrics['frame_peak_force'] = self.peak_force_frame

    def _plot_specific_events(self):
        if self.start_of_unweighting_phase is not None:
            plt.axvline(self.start_of_unweighting_phase / self.sampling_frequency,
                        color='green', label='Start of Unweighting Phase')
        if self.start_of_braking_phase is not None:
            plt.axvline(self.start_of_braking_phase / self.sampling_frequency,
                        color='yellow', label='Start of Braking Phase')
        if self.start_of_propulsive_phase is not None:
            plt.axvline(self.start_of_propulsive_phase / self.sampling_frequency,
                        color='orange', label='Start of Propulsive Phase')
        if self.peak_force_frame is not None:
            plt.axvline(self.peak_force_frame / self.sampling_frequency,
                        color='blue', linestyle='--', label='Peak Force')


class ForceTimeCurveSQJTakeoffProcessor(ForceTimeCurveTakeoffProcessor):
    """Class for computing SQJ takeoff events and metrics"""
    def __init__(self, force_series, sampling_frequency=2000, weighing_time=0.25):
        super().__init__(force_series, sampling_frequency, weighing_time=weighing_time)
        self.potential_unweighting_start = None

    def get_jump_events(self, threshold_factor_for_propulsion=10, threshold_factor_for_unweighting=5):
        """Function to get jump events for a ForceTimeCurveCMJTakeoffProcessor Class
        """
        self.start_of_propulsive_phase = get_start_of_propulsive_phase(
            force_data=self.force_series,
            sample_rate=self.sampling_frequency,
            threshold_factor=threshold_factor_for_propulsion
        )
        self.peak_force_frame = get_sqj_peak_force_event(
            force_series=self.force_series,
            start_of_propulsive_phase=self.start_of_propulsive_phase
        )
        self.potential_unweighting_start = find_potential_unweighting(
            force_data=self.force_series,
            sample_rate=self.sampling_frequency,
            threshold_factor=threshold_factor_for_unweighting
        )

    def compute_jump_metrics(self):
        """Function to compute CMJ metrics
        """
        self.jump_metrics['propulsive_peakforce_rfd_slope_between_events'] = compute_rfd(
            force_trace = self.force_series,
            window_start = self.start_of_propulsive_phase,
            window_end = self.peak_force_frame,
            sampling_frequency = self.sampling_frequency,
            method = 'average'
        )
        self.jump_metrics['propulsive_peakforce_rfd_instantaneous_average_between_events'] = compute_rfd(
            force_trace = self.force_series,
            window_start = self.start_of_propulsive_phase,
            window_end = self.peak_force_frame,
            sampling_frequency = self.sampling_frequency,
            method = 'instantaneous'
        )
        self.jump_metrics['propulsive_peakforce_rfd_instantaneous_peak_between_events'] = compute_rfd(
            force_trace = self.force_series,
            window_start = self.start_of_propulsive_phase,
            window_end = self.peak_force_frame,
            sampling_frequency = self.sampling_frequency,
            method = 'peak'
        )
        self.jump_metrics['total_net_vertical_impulse'] = integrate_area(
            time=self.time,
            signal=self.force_series_minus_bodyweight
        )
        if self.peak_force_frame > 0:
            # checks if peak_force_frame is a valid event
            self.jump_metrics['peak_force'] = self.force_series[self.peak_force_frame]
        else:
            self.jump_metrics['peak_force'] = np.nan
        self.jump_metrics['maximum_force'] = np.nanmax(self.force_series)
        self.jump_metrics['average_force_of_propulsive_phase'] = compute_average_force_between_events(
            force_trace = self.force_series,
            window_start = self.start_of_propulsive_phase,
            window_end = -1  # take the last frame
        )
        self.jump_metrics['takeoff_velocity'] = self.velocity_series[-1]
        self.jump_metrics['jump_height_takeoff_velocity'] = compute_jump_height_from_takeoff_velocity(
            self.jump_metrics['takeoff_velocity'].item()
        )
        self.jump_metrics['jump_height_net_vertical_impulse'] = compute_jump_height_from_net_vertical_impulse(
            net_vertical_impulse=self.jump_metrics['total_net_vertical_impulse'].item(),
            body_mass_kg=self.body_mass_kg
        )
        if self.start_of_propulsive_phase >= 0:
            self.jump_metrics['movement_time'] = (
                len(self.force_series) - self.start_of_propulsive_phase
            ) / self.sampling_frequency
            self.jump_metrics['propulsive_time'] = (
                len(self.force_series) - self.start_of_propulsive_phase
            ) / self.sampling_frequency
        else:
            self.jump_metrics['movement_time'] = np.nan
            self.jump_metrics['propulsive_time'] = np.nan
        self.jump_metrics['frame_start_of_propulsive_phase'] = self.start_of_propulsive_phase
        self.jump_metrics['frame_peak_force'] = self.peak_force_frame
        self.jump_metrics['frame_of_potential_unweighting_start'] = self.potential_unweighting_start

    def _plot_specific_events(self):
        if self.start_of_propulsive_phase is not None:
            plt.axvline(self.start_of_propulsive_phase / self.sampling_frequency,
                        color='orange', label='Start of Propulsive Phase')
        if self.peak_force_frame is not None:
            plt.axvline(self.peak_force_frame / self.sampling_frequency,
                        color='blue', linestyle='--', label='Peak Force')
        if self.potential_unweighting_start is not None:
            plt.axvline(self.potential_unweighting_start / self.sampling_frequency,
                        color='purple', label='Potential Unweighting Start')


class ForceTimeCurveJumpLandingProcessor:
    """Class for computing events and metrics for jump landings"""
    def __init__(self, landing_force_trace, sampling_frequency, body_weight, takeoff_velocity):
        self.landing_force_trace = landing_force_trace
        if not isinstance(self.landing_force_trace, np.ndarray):
            self.landing_force_trace = np.array(self.landing_force_trace)
        self.body_weight = body_weight
        self.sampling_frequency = sampling_frequency
        self.takeoff_velocity = takeoff_velocity
        self.body_mass_kg = self.body_weight / 9.81
        self.acceleration = (self.landing_force_trace - self.body_weight) / self.body_mass_kg
        self.velocity = compute_integral_of_signal(
            original_signal=self.acceleration,
            sampling_frequency=self.sampling_frequency,
            initial_value=self.takeoff_velocity * -1
        )
        self.landing_metrics = {}
        self.landing_metrics_dataframe = pd.DataFrame()

        self.displacement = compute_integral_of_signal(
            original_signal=self.velocity,
            sampling_frequency=self.sampling_frequency,
            initial_value=0
        )
        self.time = np.arange(
            start=0,
            stop=len(self.landing_force_trace) / self.sampling_frequency,
            step=1/self.sampling_frequency
        )
        self.end_of_landing_phase = None
        self.peak_force_frame = None
        self.kinematic_waveform_dataframe = pd.DataFrame()

    def get_landing_events(self):
        """Get the landing events"""
        self.end_of_landing_phase = get_end_of_landing_phase(
            velocity_series=self.velocity
        )
        if self.end_of_landing_phase >= 0:
            self.peak_force_frame = int(np.argmax(
                self.landing_force_trace[:self.end_of_landing_phase]
            ))
        else:
            self.peak_force_frame = int(np.argmax(self.landing_force_trace))

    def compute_landing_metrics(self):
        """Get the landing metrics"""
        events_list = [
            self.end_of_landing_phase
        ]
        if any(value is None for value in events_list):
            raise ValueError(f"The following events need to be defined first: {events_list}")
        else:
            if self.peak_force_frame > 0:
                # check if a valid frame
                self.landing_metrics['max_landing_force'] = [
                    self.landing_force_trace[
                        self.peak_force_frame
                    ]
                ]
            else:
                self.landing_metrics['max_landing_force'] = [
                    np.nan
                ]
            if self.end_of_landing_phase < 0:
                self.landing_metrics['average_landing_force'] = [np.nan]
                self.landing_metrics['landing_time'] = [np.nan]
                self.landing_metrics['landing_displacement'] = [np.nan]
            else:
                self.landing_metrics['average_landing_force'] = [
                    np.mean(self.landing_force_trace[:self.end_of_landing_phase])
                ]
                self.landing_metrics['landing_time'] = [
                    self.end_of_landing_phase / self.sampling_frequency
                ]
                self.landing_metrics['landing_displacement'] = [
                    np.min(self.displacement[:self.end_of_landing_phase])
                ]
            self.landing_metrics['landing_maxforce_rfd_slope_between_events'] = compute_rfd(
                force_trace = self.landing_force_trace,
                window_start = 0,  # first frame of landing
                window_end = self.peak_force_frame,
                sampling_frequency = self.sampling_frequency,
                method = 'average'
            )
            self.landing_metrics['landingmaxforce_rfd_instantaneous_average_between_events'] = compute_rfd(
                force_trace = self.landing_force_trace,
                window_start = 0,  # first frame of landing
                window_end = self.peak_force_frame,
                sampling_frequency = self.sampling_frequency,
                method = 'instantaneous'
            )
            self.landing_metrics['landing_maxforce_rfd_instantaneous_peak_between_events'] = compute_rfd(
                force_trace = self.landing_force_trace,
                window_start = 0,  # first frame of landing
                window_end = self.peak_force_frame,
                sampling_frequency=self.sampling_frequency,
                method = 'peak'
            )
            if self.peak_force_frame < 0:
                self.landing_metrics['max_force_frame_landing'] = np.nan
            else:
                self.landing_metrics['max_force_frame_landing'] = [
                    self.peak_force_frame
                ]

    def create_landing_metrics_dataframe(self, pid: str):
        """Create a landing metrics dataframe

        Args:
            pid (str): Participant ID

        Raises:
            ValueError: Raises if `compute_landing_metrics()` is not run before trying to create a dataframe
        """
        if not self.landing_metrics:
            raise ValueError("Must run `compute_jump_metrics()` before creating a dataframe")
        else:
            self.landing_metrics_dataframe['PID'] = [pid]
            self.landing_metrics_dataframe = pd.concat(
                [self.landing_metrics_dataframe, pd.DataFrame(self.landing_metrics, index=[0])],
                axis=1
            )

    def plot_waveform(self, waveform_type: str, title=None, savefig: bool =False, figname=None):
        """Function to plot waveforms

        Args:
            waveform_type (str): Type of waveform to plot
            title (str, optional): Title of the plot. Defaults to None.
            savefig (bool, optional): Whether the function should save the figure. Defaults to False.
            figname (str, optional): Filename for the figure. Defaults to None.
        """
        waveform_dict = {
            # waveform_type: (waveform, ylabel)
            'force': (self.landing_force_trace, 'Ground Reaction Force (N)'),
            'acceleration': (self.acceleration, 'Acceleration (m/s^2)'),
            'velocity': (self.velocity, 'Velocity (m/s)'),
            'displacement': (self.displacement, 'Displacement (m)')
        }
        if waveform_type not in waveform_dict:
            raise ValueError(f"{waveform_type} is invalid. Please specify one of: {waveform_dict.keys()}.")
        else:
            waveform, ylabel = waveform_dict[waveform_type]
            plt.plot(self.time, waveform)
            if waveform_type == 'force':
                plt.axhline(
                    self.body_weight,
                    color='red',
                    linestyle='--',
                    label=f'Bodyweight ({round(self.body_weight, 2)}N)'
                )
            if self.end_of_landing_phase is not None:
                plt.axvline(
                    self.end_of_landing_phase / self.sampling_frequency,
                    color='grey',
                    label='End of Landing Phase'
                )
            if self.peak_force_frame is not None:
                plt.axvline(
                    self.peak_force_frame / self.sampling_frequency,
                    color='blue',
                    linestyle='--',
                    label='Peak Force'
                )
                plt.legend()
            if title is not None and isinstance(title, str):
                plt.title(title)
            plt.ylabel(ylabel)
            plt.xlabel('Time (sec)')
            if not isinstance(savefig, bool):
                raise ValueError('savefig must be set to true or false (i.e., a boolean)')
            elif savefig is False:
                plt.show()
            elif savefig is True and (figname is None or not isinstance(figname, str)):
                raise ValueError('If specifying savefig, please provide a valid filename as a string')
            else:
                plt.savefig(figname, dpi=300)
                plt.close()

    def save_landing_metrics_dataframe(self, dataframe_filepath: str):
        """Function to save the jump metrics dataframe as a csv

        Args:
            dataframe_filepath (str): filename for the resulting csv

        Raises:
            ValueError: If the dataframe is empty, don't save it
        """
        if len(self.landing_metrics_dataframe) == 0:
            raise ValueError("Must run `create_dataframe` before trying to save a dataframe")
        else:
            self.landing_metrics_dataframe.to_csv(dataframe_filepath, index=False)

    def create_kinematic_dataframe(self):
        """Function to create a dataframe of the kinematic data
        """
        self.kinematic_waveform_dataframe = pd.DataFrame({
            # 'pid': [pid] * len(self.force_series),
            'acceleration': self.acceleration,
            'velocity': self.velocity,
            'displacement': self.displacement
        })

    def save_kinematic_dataframe(self, dataframe_filepath: str):
        """Function to save the kinematic dataframe

        Args:
            dataframe_filepath (str): Filepath name for the kinematic dataframe

        Raises:
            ValueError: Raises error if `create_dataframe()` was not run beforehand
        """
        if len(self.kinematic_waveform_dataframe) == 0:
            raise ValueError("Must run `create_dataframe` before trying to save a dataframe")
        else:
            self.kinematic_waveform_dataframe.to_csv(dataframe_filepath, index=False)
