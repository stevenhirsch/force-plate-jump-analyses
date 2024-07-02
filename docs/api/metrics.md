# CMJ Metrics Documentation

This document provides an overview of the functions available in the `cmj_metrics.py` file. These functions are used to compute various metrics related to countermovement jumps (CMJ), such as body weight, rate of force development (RFD), and jump height.

## `metrics.py` 

### Functions

#### `get_bodyweight`

Computes the body weight of a participant based on a static force measurement.

**Parameters**:
- `force_series (array)`: Array of force data.
- `n (int, optional)`: Number of initial frames considered as static. Defaults to 500.

**Returns**:
- `float`: Body weight in Newtons.

#### `compute_rfd`

Computes the rate of force development (RFD) during a jump between specified windows using various methods.

**Parameters**:
- `force_trace (array)`: Array of force data.
- `window_start (int)`: Start frame for RFD computation.
- `window_end (int)`: End frame for RFD computation.
- `sampling_frequency (float)`: Sampling frequency of the force plate.
- `method (str, optional)`: Method for RFD computation ('average', 'instantaneous', 'peak'). Defaults to 'average'.

**Returns**:
- `float`: RFD in Newtons per second.

#### `compute_jump_height_from_takeoff_velocity`

Computes jump height from takeoff velocity.

**Parameters**:
- `takeoff_velocity (float)`: Takeoff velocity in meters per second.

**Returns**:
- `float`: Jump height in meters.

#### `compute_jump_height_from_velocity_series`

Computes jump height from a velocity series.

**Parameters**:
- `velocity_series (array)`: Array of movement velocities.

**Returns**:
- `float`: Jump height in meters.

#### `compute_jump_height_from_net_vertical_impulse`

Computes jump height from the net vertical impulse of a CMJ.

**Parameters**:
- `net_vertical_impulse (float)`: Net vertical impulse of the jump.
- `body_mass_kg (float)`: Body mass of the participant in kilograms.

**Returns**:
- `float`: Jump height in meters.

#### `compute_average_force_between_events`

Computes the average force between two events.

**Parameters**:
- `force_trace (array)`: Array of force data.
- `window_start (int)`: Start frame for average force computation.
- `window_end (int)`: End frame for average force computation.

**Returns**:
- `float`: Average force in Newtons.