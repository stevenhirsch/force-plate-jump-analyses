# force-plate-jump-analyses

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

## Overview
**force-plate-jump-analyses** is a set of functions and tools for analyzing (countermovement) jump data collected using force plates. The package provides comprehensive analyses to help in understanding various aspects of jump performance and mechanics and helps with batch processing jump trials.

## Features
- Batch processing of CMJ data from `.txt` files exported from force plates.
- Signal processing and filtering
- Automated data cropping, events, metrics, and visualization tools.
- Easy-to-use functions for detailed jump performance insights.
- Extensible framework for additional analysis and custom data processing.

## Getting Set Up to Run the Code
### Prerequisites
- **Anaconda** or **Miniconda**: Install [Anaconda](https://anaconda.org) or [Miniconda](https://docs.anaconda.com/miniconda/) on your machine.
- **Python**: Ensure Python 3.8 or higher is installed.

### Installation
1. Clone the repository to your local machine:
    ```
    git clone https://github.com/your-username/force-plate-jump-analyses.git
    cd force-plate-jump-analyses
    ```

2. Create the environment using the provided `env.yml` file:
    ```
    conda env create -f env.yml
    ```

3. Activate the environment:
    ```
    conda activate jumpmetrics
    ```

4. Optionally, for extra pacakges used for local development, run:
    ```
    conda env update -f local_env.yml
    ```

## Building the Package
If you want to build the package for installation, you can do so using `python -m build`. Follow these steps to create a package:

1. **Install `build` package**:
    ```
    pip install build
    ```

2. **Build the package**:
    Navigate to the root directory where `setup.py` is located, and run:
    ```
    python -m build
    ```

    This command will generate distribution archives (`.tar.gz` and `.whl` files) in the `dist` directory.

3. **Check the build**:
    Ensure the package was created successfully by checking the `dist` directory for the generated files.

    ```
    ls dist/
    ```


### Installing the Built Package
To install the built package locally, use `pip` with the path to the `.whl` file:

`pip install dist/jumpmetrics-0.1.0-py3-none-any.whl`

<!-- 
## Detailed Documentation
For a complete guide on available functions and their usage, please refer to the [Documentation](./docs).

## Contributing
We welcome contributions! Please read our [Contributing Guide](./CONTRIBUTING.md) to learn how you can help improve this project.
-->

## Batch Processing Data
Examples of how to batch process data are found in `study_1_batch_process.py` and `study_2_batch_process.py`. You can run eithet file with a command such as:
```
python study_1_batch_process.py
```

## Reporting Issues
If you encounter any issues, please open an [issue on GitHub](https://github.com/stevenhirsch/force-plate-jump-analyses/issues).

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contact
For any questions or further information, please contact me on my [website](https://www.stevenhirsch.ca/contact/)
<!--
## Acknowledgements
Special thanks to everyone who has contributed to the development of this package.

## Additional Resources
- [Force Plate Analysis Fundamentals](https://example.com/force-plate-analysis)
- [CMJ Performance Analysis Techniques](https://example.com/cmj-performance)

## Changelog
For detailed information on recent updates, check the [Changelog](./CHANGELOG.md).
-->