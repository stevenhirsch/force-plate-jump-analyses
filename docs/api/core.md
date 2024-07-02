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
