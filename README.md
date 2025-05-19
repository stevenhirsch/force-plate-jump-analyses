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

### Option 1: Direct Installation from GitHub
Ensure that you have `pip` installed in your python environment. Then, just run:
```
pip install git+https://github.com/stevenhirsch/force-plate-jump-analyses.git 
```

### Option 2: Build and Install from a Local Copy
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

### Option 3: Development Setup with Conda
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

## Detailed Documentation
For a complete guide on available functions and their usage, please refer to the [Documentation](./docs/index.md).
<!-- 
## Contributing
We welcome contributions! Please read our [Contributing Guide](./CONTRIBUTING.md) to learn how you can help improve this project.
-->

## Batch Processing Data
Examples of how to batch process data for a study are found in `study_1_batch_process.py`,  `study_2_batch_process.py`, and `study_3_batch_process.py`. You can run either file with a command such as:
```
python study_1_batch_process.py
```

## Reporting Issues
If you encounter any issues, please open an [issue on GitHub](https://github.com/stevenhirsch/force-plate-jump-analyses/issues).

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contact
For any questions or further information, please contact me on my [website](https://www.stevenhirsch.ca/contact/) or via [LinkedIn](https://www.linkedin.com/in/steven-m-hirsch/).