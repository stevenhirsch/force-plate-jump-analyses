"""IO functions"""
import logging  # for logging errors
import pandas as pd
import numpy as np

NOT_FOUND = -10 ** 2


def load_cropped_force_data(filepath: str, freq=None) -> pd.Series:
    """Load already cropped force data from a filepath that contains several filtered
    force series. If a freq is not specified, load the raw data. Otherwise, load the
    dataframe with the corresponding filter cutoff frequency.

    Args:
        filepath (str): Filepath of the force data of interest
        freq optional): Cutoff frequency to examine. Defaults to None.

    Returns:
        pd.DataFrame: Force trace data series
    """
    try:
        df = pd.read_csv(filepath, delimiter='\t', skiprows=[0, 2, 3, 4])
    except FileNotFoundError:
        logging.error("The file at %s was not found.", filepath)
        return None
    except pd.errors.ParserError as e:
        logging.error("Parsing error while reading the file: %s", e)
        return None
    # except Exception as e:
    #     logging.error("An unexpected error occurred while reading the file: %s", e)
    #     return None
    if freq is None:
        colname = 'FZsum'
    else:
        colname = f'FZsum_{freq}Hz'
    if colname not in df.columns:
        logging.warning(
            "Column '%s' not found in the dataframe. Available columns: %s", colname, df.columns.tolist()
        )
        return None

    try:
        force_data = df[colname]
    except KeyError:
        logging.warning("Key %s not found in the dataframe.", colname)
        return None
    # except Exception as e:
    #     logging.error(f"An unexpected error occurred while accessing the column: {e}")
    #     return None

    return force_data


def load_raw_force_data(filepath: str) -> pd.DataFrame:
    """Loads a raw force data file into a Pandas DataFrame

    Args:
        filepath (str): Filepath of the force data of interest

    Returns:
        pd.DataFrame: Dataframe of the force data
    """
    try:
        df = pd.read_csv(filepath, delimiter='\t', skiprows=[0, 2, 3, 4])
    except FileNotFoundError:
        logging.error("The file at %s was not found.", filepath)
        return None
    except pd.errors.ParserError as e:
        logging.error("Parsing error while reading the file: %s", e)
        return None
    # except Exception as e:
    #     logging.error("An unexpected error occurred while reading the file: %s", e)
    #     return None

    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    return df


def load_raw_force_data_with_no_column_headers(filepath: str) -> pd.DataFrame:
    """Function to load raw force data directly from Netforce
    that does not contain column headers

    Args:
        filepath (str): Filepath of the raw force data

    Returns:
        pd.DataFrame: Dataframe of the force data
    """
    try:
        # Read the file, skip the first blank line, and use comma as delimiter
        df = pd.read_csv(filepath, delimiter=',', header=None, skiprows=1)

        # Strip whitespace from all cells
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Define column names
        cols = ['FX1', 'FY1', 'FZ1', 'MX1', 'MY1', 'MZ1', 'FX2', 'FY2', 'FZ2', 'MX2', 'MY2', 'MZ2']

        # Assign column names
        df.columns = cols

        return df
    except FileNotFoundError:
        logging.error("The file at %s was not found.", filepath)
        return None
    except pd.errors.ParserError as e:
        logging.error("Parsing error while reading the file: %s", e)
        return None


def sum_dual_force_components(
    force_df: pd.Series, component_1: str ='FZ1', component_2: str ='FZ2'
) -> pd.Series:
    """Sums two components of the force plate into a single component

    Args:
        force_df (pd.Series): Force Trace
        component_1 (str, optional): Component 1. Defaults to 'FZ1'.
        component_2 (str, optional): Component 2. Defaults to 'FZ2'.

    Returns:
        pd.Series: Summed force series
    """
    summed_force_component = force_df[component_1] + force_df[component_2]
    return summed_force_component


def find_first_frame_where_force_exceeds_threshold(force_trace: pd.Series, threshold: float) -> int:
    """Find first frame where the force trace exceeds a threshold. This function is mainly used to
    help find the takeoff event since a recording may start with someone not yet on the force plate. Therefore,
    we may look for the first frame where the force exceeds some threshold and then use `find_takeoff_frame()`
    only after the first frame where force exceeded some threshold.

    Args:
        force_trace (pd.Series): Force trace
        threshold (float): Minimum threshold

    Returns:
        int: Frame where force exceeds threshold.
    """
    # Can set a nice threshold if you have participants' body mass a priori, for example
    frames_where_force_exceeds_threshold = np.where(force_trace >= threshold)[0]
    if len(frames_where_force_exceeds_threshold) == 0:
        logging.warning(
            """No frames detected that exceeded the specified threshold of %s, returning -1""",
            threshold
        )
        return -1
    first_frame_where_force_exceeds_threshold = frames_where_force_exceeds_threshold[0]
    return first_frame_where_force_exceeds_threshold


def find_frame_when_off_plate(
    force_trace: pd.Series, sampling_frequency: float,
    flight_time_threshold: float = 0.25, force_threshold: float = 10
) -> int:
    """Find the next frame when the person is off the force plate.
    Used to determine takeoff frame of a vertical jump, for example.

    Args:
        force_trace (pd.Series): Force trace of the vertical jump
        sampling_frequency (float): Sampling frequency of the force plate
        flight_time_threshold (float, optional): Threshold for defining a "flight" phase. Defaults to 0.25s.
        force_threshold (float, optional): Threshold to determine if someone is "off" the plate. Defaults to 10N.

    Returns:
        int: Frame corresponding to takeoff
    """
    # Convert time threshold to number of samples
    sample_threshold = int(flight_time_threshold * sampling_frequency)

    # Initialize counter for consecutive low-force samples
    consecutive_low_force_samples = 0

    for index, force in force_trace.items():
        if abs(force) < force_threshold:
            consecutive_low_force_samples += 1
            if consecutive_low_force_samples >= sample_threshold:
                # We've found the takeoff point, return the index of the start of this sequence
                return int(force_trace.index[force_trace.index.get_loc(index) - sample_threshold + 1])
        else:
            consecutive_low_force_samples = 0

    # If we get here, no takeoff was detected
    return NOT_FOUND


def get_n_seconds_before_takeoff(
    force_trace: pd.Series, sampling_frequency: float, takeoff_frame: int, n: float = 2.
) -> pd.Series:
    """Function to obtain the first n seconds before takeoff for a jump.

    Args:
        force_trace (pd.Series): Force trace
        sampling_frequency (float): Sampling frequency of the force plate
        takeoff_frame (int): Frame corresponding to takeoff
        n (float, optional): Number of seconds before takeoff to keep. Defaults to 2..

    Returns:
        pd.Series: Cropped force_trace to only n seconds before takeoff
    """
    # Calculate the number of samples for n seconds
    n_samples = int(n * sampling_frequency)

    if takeoff_frame is None:
        # return the force trace data since the person was never detected
        # to be off the plate
        return force_trace
    else:
        # Get the integer location of the takeoff frame
        takeoff_loc = force_trace.index.get_loc(takeoff_frame)

        # Calculate the start location
        start_loc = max(0, takeoff_loc - n_samples)

        # Slice the force_trace
        return force_trace.iloc[start_loc:takeoff_loc].reset_index(drop=True)


def find_landing_frame(force_series, n=30, threshold_value=20):
    """
    Find the landing frame in a vertical jump where the force
    remains above the threshold for a specified number of frames.

    Parameters:
    force_series (list or array): The series of force values recorded over time.
    n (int): The number of consecutive frames that must be above the threshold. Defaults to 30.
    threshold_value (float): The force threshold value. Defaults to 20.

    Returns:
    int: The index of the landing frame where the force first meets the criteria. Returns -1 if no such frame is found.
    """
    total_frames = len(force_series)

    for i in range(total_frames - n + 1):
        # Check if the next 'n' frames are all above the threshold_value
        if all(force >= threshold_value for force in force_series[i:i + n]):
            return i  # Return the first frame where this condition is met

    logging.warning(' No landing frame found.')
    return -1  # Return -1 if no such frame is found

def get_end_of_landing_phase(velocity_series):
    """Function to get the end of the landing phase"""
    velocities_greater_than_zero = np.where(velocity_series >= 0)[0]
    if len(velocities_greater_than_zero) > 0:
        return velocities_greater_than_zero[0]
    else:
        return -1
