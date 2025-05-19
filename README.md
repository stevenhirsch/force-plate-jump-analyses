# force-plate-jump-analyses

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

## Overview

The `JumpMetrics` package is a completely free, open-source toolkit for analyzing force plate data during countermovement and squat (pause) jumps. The package provides comprehensive analyses to help in understanding various aspects of jump performance and mechanics and helps with batch processing jump trials.

## Features
- Batch processing of countermovement jump and squat jump data from `.txt` files exported from force plates (e.g., NetForce).
- Signal processing and filtering.
- Automated data cropping, events, metrics, and visualization tools.
- Easy-to-use functions for detailed jump performance insights.
- Extensible framework for additional analysis and custom data processing.

## Installation

### Prerequisites
- **Python**: Ensure Python 3.10 or higher is installed.
    - **Anaconda** or **Miniconda**: Install [Anaconda](https://anaconda.org) or [Miniconda](https://docs.anaconda.com/miniconda/) on your machine.

### Option 1: Install via PyPI

*NOTE:* Pending upload to PyPI

Simply run:
```
pip install jumpmetrics
```

### Option 2: Direct Installation from GitHub (may be slow due to repo size)
Ensure that you have `pip` installed in your python environment. Then, just run:
```
pip install git+https://github.com/stevenhirsch/force-plate-jump-analyses.git 
```

### Option 3: Build and Install from a Local Copy
1. Clone the repository:
```
git clone https://github.com/stevenhirsch/force-plate-jump-analyses.git
cd force-plate-jump-analyses
```

2. Build the package
```
pip install build
python -m build
```

This command will generate distribution archives (`.tar.gz` and `.whl` files) in the `dist` directory.

3. Install the wheel file (adjust the filename to whichever version you have built)
```
pip install dist/jumpmetrics-0.1.0-py3-none-any.wh
```

After going through these steps, try running:
```
python test_install.py
```

for additional verification that the package installed correctly.

### Option 4: Development Setup with Conda
1. Clone the repository:
```
git clone https://github.com/stevenhirsch/force-plate-jump-analyses.git
cd force-plate-jump-analyses
```

2. Create and activate the conda environment
```
conda env create -f env.yml
conda activate jumpmetrics
conda env update local_env.yml
```

## Data Processing
The following code snippet should help to generally showcase how one could get started quickly with `jumpmetrics` for calculating takeoff metrics:
```python
from jumpmetrics.core.core import ForceTimeCurveCMJTakeoffProcessor
from jumpmetrics.core.io import (
    load_raw_force_data_with_no_column_headers, sum_dual_force_components,
    find_first_frame_where_force_exceeds_threshold,
    find_frame_when_off_plate, get_n_seconds_before_takeoff
)
from jumpmetrics.signal_processing.filters import butterworth_filter

# Load a force dataset
tmp_force_df = load_raw_force_data_with_no_column_headers(filepath)
# Sum the vertical force components from a data collection that uses dual force plates
# Note that the goal is to simply just get a force waveform, and these are helper functions to do so
# However, you could use your own custom code to obtain a force trace for processing
full_summed_force = sum_dual_force_components(tmp_force_df)
# Helper function to help narrow force series to identify when someone is on or off the plate
frame = find_first_frame_where_force_exceeds_threshold(
    force_trace=full_summed_force,
    threshold=1000
)
# Helper function to find when someone is off the plate. Can be used to determine the moment of takeoff
takeoff_frame = find_frame_when_off_plate(
    force_trace=full_summed_force.iloc[frame:],
    sampling_frequency=2000
)
# Cropped force trace provides just the first n seconds before takeoff for processing
 cropped_force_trace = get_n_seconds_before_takeoff(
            force_trace=full_summed_force,
            sampling_frequency=2000,
            takeoff_frame=takeoff_frame,
            n=TIME_BEFORE_TAKEOFF
)
# Filtering the force trace data (optional step)
filtered_force_series = butterworth_filter(
    arr=cropped_force_trace,
    cutoff_frequency=50,
    fps=2000,
    padding=2000
)
# Instantiative the takeoff processor class
CMJ = ForceTimeCurveCMJTakeoffProcessor(
    force_series=filtered_force_series,
    sampling_frequency=2000
)
# Get the jump events
CMJ.get_jump_events()
# Get the jump metrics
CMJ.compute_jump_metrics()
# Create a jump metric dataframe
CMJ.create_jump_metrics_dataframe()
# Create a kinematic data dataframe
CMJ.create_kinematic_dataframe()
# Plot the waveform data
CMJ.plot_waveform(
    waveform_type='force',
    title='Testing',
    savefig=True,
    figname=os.path.join('force.png')
)
# Save the dataframe
CMJ.save_kinematic_dataframe(
    dataframe_filepath=os.path.join(pid_data_dir, 'kinematic_data.csv')
)
```

Processing an entire jump (takeoff and landing) can be done using the following code example OR building your own wrapper functions using the landing and takeoff classes.
```python
tmp_force_df = load_raw_force_data_with_no_column_headers(filepath)
full_summed_force = sum_dual_force_components(tmp_force_df)
results_dict = process_jump_trial(
    full_force_series=full_summed_force,
    sampling_frequency=2000,
    jump_type='countermovement',
    weighing_time=0.25,
    pid='test1',
    threshold_for_helping_determine_takeoff=1000,
    lowpass_filter=True,
    lowpass_cutoff_frequency=26.64,
    compute_jump_height_from_flight_time=True
)
```

Examples of how to batch process data for a study are found in `study_1_batch_process.py`,  `study_2_batch_process.py`, and `study_3_batch_process.py`. You can run either file with a command such as:
```
python study_1_batch_process.py
```

For a complete guide on available functions and their usage, please refer to the [Documentation](./docs/index.md).

## Reporting Issues
If you encounter any issues, please open an [issue on GitHub](https://github.com/stevenhirsch/force-plate-jump-analyses/issues).

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contact
For any questions or further information, please contact me on my [website](https://www.stevenhirsch.ca/contact/) or via [LinkedIn](https://www.linkedin.com/in/steven-m-hirsch/).