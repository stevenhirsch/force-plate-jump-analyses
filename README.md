# force-plate-jump-analyses
Functions and Analyses for CMJ data on force plates.

# Getting Set Up to Run the Code
Install [Anaconda](https://anaconda.org), or preferentially [Miniconda](https://docs.anaconda.com/miniconda/), on your machine. Then, use the env.yml file to create your environment.

To create the environment using your command line, run `conda create env -f env.yaml`. This will create an environment named `jump-analysis`. Run `conda activate jump-analysis` to get started.

# Running the Functions
To run each study's analysis, navigate to the repository and look for the file named `batch_process.py`. This file is designed to batch process all the `.txt` files in the study's directory.
