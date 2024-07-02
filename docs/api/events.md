# CMJ Events Documentation

This document provides an overview of the functions available in the `cmj_events.py` file. These functions are designed to analyze various phases of a countermovement jump (CMJ) using force, velocity, and displacement data.

## `events.py`

### Functions

#### `find_unweighting_start`

Identifies the start of the unweighting phase in a countermovement jump using force data.

**Parameters**:
- `force_data (array)`: Force series data.
- `sample_rate (float)`: Sampling rate of the force plate.
- `quiet_period (int, optional)`: Duration (in seconds) of the initial quiet stance period. Defaults to 1 second.
- `threshold_factor (float, optional)`: Number of standard deviations below the mean force used to determine the start of unweighting. Defaults to 5.
- `window_size (float, optional)`: Size of the window (in seconds) for the Savitzky-Golay filter. Defaults to 0.2 seconds.
- `duration_check (float, optional)`: Number of seconds to check if the person is unweighting. Defaults to 0.1 seconds.

One thing to note is that the method of finding this unweighting phase is based largely on the method outlined in [Owen et al. (2014)](https://journals.lww.com/nsca-jscr/fulltext/2014/06000/development_of_a_criterion_method_to_determine.8.aspx), which was highlighted by [McMahon et al. (2018)](https://journals.lww.com/nsca-scj/fulltext/2018/08000/Understanding_the_Key_Phases_of_the.10.aspx?casa_token=ebRHgNsbZ8oAAAAA:zhLpS7rrORCZWHesIP2TzfvEHVXoHMhKL9xsfE-p4Qk73EXINHbQd1j2s3oK8TCN_DZyJuBgP8_Wurzh6VWSfwTsEg)

**Returns**:
- `int`: Frame number corresponding to the start of the unweighting phase.

#### `get_start_of_unweighting`

Finds the start of the unweighting phase using velocity data.

**Parameters**:
- `velocity_series (array)`: Array of velocity data.

**Returns**:
- `int`: Frame number corresponding to the start of the unweighting phase.

#### `get_start_of_concentric_phase_using_velocity`

Determines the start of the concentric phase of a countermovement jump using velocity data.

**Parameters**:
- `velocity_series (array)`: Array of velocity data.

**Returns**:
- `int`: Frame number corresponding to the start of the concentric phase.

#### `get_start_of_braking_phase_using_velocity`

Identifies the start of the braking phase using velocity data.

**Parameters**:
- `velocity_series (array)`: Array of velocity data.

**Returns**:
- `int`: Frame number corresponding to the start of the braking phase.

#### `get_start_of_propulsive_phase_using_displacement`

Finds the start of the propulsive phase using displacement data.

**Parameters**:
- `displacement_series (array)`: Array of displacement data.

**Returns**:
- `int`: Frame number corresponding to the start of the propulsive phase.

#### `get_peak_force_event`

Identifies the peak force event during the propulsive phase using force data.

**Parameters**:
- `force_series (array)`: Array of force data.
- `start_of_propulsive_phase (int)`: Frame number corresponding to the start of the propulsive phase.

**Returns**:
- `int`: Frame number corresponding to the peak force.