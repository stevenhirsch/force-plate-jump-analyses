# Overview of ForceTimeCurveCMJTakeoffProcessor Class

The `ForceTimeCurveCMJTakeoffProcessor` class is designed to compute and analyze various events and metrics associated with a countermovement jump (CMJ) based on force-time data. This class processes the force-time curve to extract key events and compute relevant kinematic metrics, such as acceleration, velocity, and displacement. It provides methods for identifying critical phases in the jump, computing various performance metrics, and generating data visualizations.

## Key Features
- **Event Detection**: Automatically identifies critical events during a CMJ, such as the start of the unweighting phase, braking phase, and propulsive phase, as well as the peak force event.
- **Metric Computation**: Computes a comprehensive set of metrics, including rate of force development (RFD), net vertical impulse, average and peak forces, and jump height.
- **Kinematic Data Generation**: Calculates kinematic series for acceleration, velocity, and displacement from the force-time data.
- **Data Management**: Supports creation and saving of metrics and kinematic data into structured dataframes for further analysis.
- **Visualization**: Provides methods to plot and visualize the force-time curve and other waveforms, including key events.

## Detailed Class Structure

### Initialization

The class is initialized with a force series and an optional sampling frequency. The initialization process involves:

- Converting the force series into a pandas Series.
- Calculating the body weight and body mass.
- Generating kinematic series for acceleration, velocity, and displacement.
- Preparing empty data structures for storing computed metrics and waveform data.

### Key Methods

1. `get_jump_events`:

- Identifies key events in the CMJ, including the start of the unweighting, braking, and propulsive phases, and the peak force event.
- Utilizes force, velocity, and displacement data to accurately detect these events.

2. `compute_jump_metrics`:

- Calculates various performance metrics using the identified events.
- Metrics include RFD during different phases, net vertical impulse, average forces, jump height, and other temporal and spatial characteristics.

3. `create_jump_metrics_dataframe`:

- Creates a dataframe from the computed metrics, allowing for structured data storage.

4. `plot_waveform`:

- Plots specified waveforms, including force, acceleration, velocity, and displacement.
- Marks critical events on the plot for visual analysis.
- Supports saving the plot to a file.

5. `save_jump_metrics_dataframe`:

- Saves the computed metrics dataframe to a CSV file for external use.

6. `create_kinematic_dataframe`:

- Creates a dataframe for kinematic data, including acceleration, velocity, and displacement series.

7. `save_kinematic_dataframe`:

- Saves the kinematic data to a CSV file.
- Ensures the dataframe has been populated before saving.

## Exported Metrics: Peak Force and Net Vertical Impulse

`compute_jump_metrics` exports several force- and impulse-related keys whose exact definitions are
not obvious from their names alone:

- **`peak_force`** is the frame of the *first prominent peak* (see `get_peak_force_event` in
  [`events.md`](events.md)) found at or after the start of the braking phase, not the global maximum
  of the trace. On a bimodal force-time curve, this correctly captures an earlier, higher peak that
  precedes the low position. If no peak clears the prominence threshold, `peak_force` is `np.nan`
  (a warning is logged).
- **`maximum_force`** is the global maximum of the force series (`np.nanmax`). Use this if you want
  the single highest recorded force value regardless of peak shape; use `peak_force` if you want the
  force-plate-community definition of "peak force" as a genuine local maximum of the curve.
- **`frame_propulsive_peak_force`** is a second, independent peak-force event, searched starting at
  the start of the propulsive phase (the low position) rather than the start of braking. It exists
  only to anchor the three `propulsive_peakforce_rfd_*` metrics, which by definition describe the
  propulsive phase; it is not used for the `peak_force` metric. It can be `NOT_FOUND` (`-100`) more
  often than `frame_peak_force`, since a jump can have a monotonic rise from the low position to
  takeoff with no additional prominent peak in between — in that case the three
  `propulsive_peakforce_rfd_*` metrics are `np.nan`.
- **Net vertical impulse windows.** All four impulse metrics integrate `force - bodyweight` (the
  trapezoidal rule) and are related by the impulse-momentum theorem (impulse `= body_mass_kg * Δv`
  over the same window):
  - `braking_net_vertical_impulse`: start of braking phase → low position (start of propulsive
    phase).
  - `propulsive_net_vertical_impulse`: low position → takeoff (the last frame of the trace).
  - `braking_to_propulsive_net_vertical_impulse`: start of braking phase → takeoff — the sum of the
    two windows above.
  - `total_net_vertical_impulse`: the entire trace (quiet stance → takeoff).

  The braking and propulsive windows deliberately share the low-position sample as their common
  boundary, so `braking_net_vertical_impulse + propulsive_net_vertical_impulse` equals
  `braking_to_propulsive_net_vertical_impulse` exactly (to floating-point precision), not
  approximately.

## Example Usage
To use the `ForceTimeCurveCMJTakeoffProcessor`, instantiate the class with a force series and a sampling frequency. Call `get_jump_events` to identify the key events, followed by `compute_jump_metrics` to calculate the metrics. Use `create_jump_metrics_dataframe` and `save_jump_metrics_dataframe` to save the results for further analysis. You can also plot the waveform data using `plot_waveform`.

```
# Example usage
force_series = pd.Series([...])  # your force data here
processor = ForceTimeCurveCMJTakeoffProcessor(force_series, sampling_frequency=2000)
processor.get_jump_events()
processor.compute_jump_metrics()
processor.create_jump_metrics_dataframe("Participant1")
processor.save_jump_metrics_dataframe("metrics.csv")
processor.plot_waveform("force", title="Force-Time Curve")
```

This class provides a comprehensive toolkit for analyzing CMJ data, making it invaluable for researchers and practitioners in sports science and biomechanics.
