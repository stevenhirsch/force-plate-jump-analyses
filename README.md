# JumpMetrics

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
- **pixi**: Install [pixi](https://pixi.sh/latest/) on your machine.

### Option 1: Install via PyPI

*NOTE:* Pending upload to PyPI- This has not been completed yet, so skip Option 1 for now.

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

#### Method A: Using pixi (Recommended)
1. Clone the repository:
```bash
git clone https://github.com/stevenhirsch/force-plate-jump-analyses.git
cd force-plate-jump-analyses
```

2. Set up the environment and build:
```bash
pixi install
pixi shell --feature dev
pixi run build
```

3. Install the built package:
```bash
pip install dist/jumpmetrics-0.1.0-py3-none-any.whl
```

#### Method B: Using pip directly
1. Clone the repository:
```bash
git clone https://github.com/stevenhirsch/force-plate-jump-analyses.git
cd force-plate-jump-analyses
```

2. Build the package:
```bash
pip install build
python -m build
```

3. Install the wheel file (adjust the filename to whichever version you have built):
```bash
pip install dist/jumpmetrics-0.1.0-py3-none-any.whl
```

After going through these steps, verify the installation:
```bash
python test_install.py
```

### Option 4: Development Setup with pixi
1. Clone the repository:
```bash
git clone https://github.com/stevenhirsch/force-plate-jump-analyses.git
cd force-plate-jump-analyses
```

2. Create and activate the development environment:
```bash
pixi install
pixi shell --feature dev
```

This will set up a development environment with all necessary dependencies including:
- Core dependencies: pandas, matplotlib, scipy, scikit-learn
- Development tools: pytest, mypy, flake8, jupyter, ipython
- Visualization libraries: plotly, seaborn
- Build tools and utilities

The development environment provides additional tools for contributing to the project.

### Available pixi Tasks

The project includes several predefined tasks that can be run using `pixi run <task-name>`:

#### Main Tasks (Default Environment)
- `verify`: Run installation verification test
- `batch_process_study_1`: Process data for study 1
- `batch_process_study_2`: Process data for study 2
- `batch_process_study_3`: Process data for study 3

#### Development Tasks
- `build`: Build the package distribution
- `lint`: Run flake8 linting on source code
- `clean`: Remove build artifacts and cache files
- `test`: Run pytest test suite
- `typecheck`: Run mypy type checking

Example usage:
```bash
# Run tests
pixi run test

# Build the package
pixi run build

# Run linting
pixi run lint
```

### Option 5: Using Docker
For users who prefer Docker or desire a reproducible environment across different systems, we also provide a Dockerfile to easily set up and run jumpmetrics. This Dockerfile provides a way for you to separate your environment (the Docker image) from the analysis code (mounted scripts).

1. Build the Docker image:
```bash
docker build -t jumpmetrics .
```

You can also add tags to this build:
```bash
docker build -t jumpmetrics:v1.0 .
```

2. Create your analysis script. See an example at [docker_example/scripts/docker_example.py](/docker_example/scripts/docker_example.py).

3. Run your analysis with Docker:
```bash
docker run -it --rm \
  -v /path/to/your/scripts:/scripts \
  -v /path/to/your/input/data:/data/input \
  -v /path/to/your/output:/data/output \
  jumpmetrics python /scripts/my_analysis.py
```

The Docker container provides a pre-configured environment with jumpmetrics installed. You can:
- Mount your analysis scripts using `-v /path/to/your/scripts:/scripts`
- Mount your input data using `-v /path/to/your/input/data:/data/input`
- Mount an output directory using `-v /path/to/your/output:/data/output`

Directory structure example:
```
~/my_jump_analysis/
  ├── scripts/
  │   └── my_analysis.py    # Your analysis script
  ├── input/
  │   └── F02_CTRL1.txt    # Your force plate data
  └── output/              # Results will appear here
      ├── jump_metrics.csv
      ├── kinematic_data.csv
      └── force_curve.png
```

With the current repository structure, you could therefore run:
```bash
docker run -it --rm \
  -v ./docker_example/scripts:/scripts \
  -v ./docker_example/input:/data/input \
  -v ./docker_example/output:/data/output \
  jumpmetrics python /scripts/docker_example.py
```

to test this output (check out [/docker_example/input](/docker_example/input/) and [/docker_example/output](/docker_example/output/) to see what this looks like).

This setup allows you to:
1. Keep your analysis scripts separate from the package
2. Modify your analysis without rebuilding the Docker image
3. Run different analyses using the same container
4. Share your analysis scripts while ensuring they run in the same environment

If you want to explore this environment interactively, run:
```bash
docker run -it jumpmetrics bash
```

## Data Processing
The following code snippet should help to generally showcase how one could get started quickly with `jumpmetrics` for calculating takeoff metrics:
```python
from jumpmetrics.core.processors import ForceTimeCurveCMJTakeoffProcessor
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

### Example Output

The following is an example of the events that can be detected during the takeoff phase of a vertical jump:

![Example countermovement jump force-time trace with events detected during the takeoff phase.](/analyses/study_1/figures/F02/CTRL1/literature_cutoff/force.png)


These events are the foundation for all computed metrics. For a full list of metrics for both the takeoff and landing phases, please refer to the tables in ![](/paper/paper.md).

## Assumptions In this Package

Please read the paper for full details. A summary of a portion of the paper is provided below:

> To compute the relevant events and metrics, there are specific methods that a user should adhere to that are outlined in previous literature. These methods underpin the assumptions required to collect and process data with the code provided in this package. First, the jumper must stand still (i.e., minimizing swaying or any other body movements) at the start of the data collection and for at least 1 second before starting the initiation of the jump. This quiet standing is used to calculate one's bodyweight, and bodyweight is used for subsequent acceleration, velocity, and displacement calculations used for event detections (as well as for computing net vertical impulse). The default setting in this package is currently to use the first 0.4 seconds of the trial to compute bodyweight (as this was found to work well for previous analyses), but users can tune this parameter themselves for their own data collections depending on the length of the quiet standing at the start of the data collection. For a countermovement jump, the functions in this package also require the person to perform one continuous downwards and upwards motion during the jump; any pausing may negatively impact the event detection algorithms provided. In contrast, for the squat jump the default parameter for identifying the start of the propulsive phase expects at least a 1 second pause. In practice, previous research has outlined a pause should be approximately 3 seconds. The functions provided in `JumpMetrics` permit the user to select a different minimum pause to assume if the default of 1 second is not appropriate for their research.

## Reporting Issues
If you encounter any issues, please open an [issue on GitHub](https://github.com/stevenhirsch/force-plate-jump-analyses/issues).

## Contributing
We welcome contributions from researchers, practitioners, and developers! Please see our [Contributing Guidelines](docs/development/contributing.md) for details on how to get started.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contact
For any questions or further information, please contact me on my [website](https://www.stevenhirsch.ca/contact/) or via [LinkedIn](https://www.linkedin.com/in/steven-m-hirsch/).