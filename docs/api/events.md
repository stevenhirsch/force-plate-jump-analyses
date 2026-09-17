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

One thing to note is that the method of finding this unweighting phase is based largely on the method outlined in [Owen et al. (2014)](https://journals.lww.com/nsca-jscr/fulltext/2014/06000/development_of_a_criterion_method_to_determine.8.aspx), which was highlighted by [McMahon et al. (2018)](https://journals.lww.com/nsca-scj/fulltext/2018/08000/Understanding_the_Key_Phases_of_the.10.aspx?casa_token=ebRHgNsbZ8oAAAAA:zhLpS7rrORCZWHesIP2TzfvEHVXoHMhKL9xsfE-p4Qk73EXINHbQd1j2s3oK8TCN_DZyJuBgP8_Wurzh6VWSfwTsEg).

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

Identifies the peak force event of the jump, defined as the first prominent peak in the force series
after `search_start_frame`. Peak detection uses `find_peaks` from `scipy`; see scipy's documentation
for more information about prominence. Note that `find_peaks` cannot flag the first sample of the
searched slice as a peak (a peak needs a neighbour on both sides), so a genuine peak located exactly
at `search_start_frame` will not be detected there.

For a countermovement jump, `search_start_frame` should be the start of the braking phase. Peak force
can occur before the low position on bimodal force profiles, so anchoring the search at the start of
the propulsive phase misses the true peak on those jumps. Braking onset is where force rises back
through bodyweight, so the first prominent peak after it is the first genuine force peak of the jump,
while peaks during quiet stance and within the unweighting dip are excluded.

Note that this returns the *first* prominent peak in the search window, not the largest one. Use the
`maximum_force` metric for the global maximum of the trace.

**Parameters**:
- `force_series (array)`: Array of force data.
- `search_start_frame (int)`: Frame to begin searching for a peak from. For a countermovement jump
  this should be the start of the braking phase. If `None` or negative, the entire force series is
  searched and a warning is logged.
- `prominence (float, optional)`: Minimum prominence for a peak to be detected. Defaults to `50`.

**Returns**:
- `int`: Frame number corresponding to the peak force, or `NOT_FOUND` (`-100`) if no peak of the
  required prominence was detected. A warning is logged in that case, and downstream metrics that
  depend on this event return `np.nan`.