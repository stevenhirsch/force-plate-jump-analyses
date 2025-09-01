# Contributing to JumpMetrics

Thank you for your interest in contributing to JumpMetrics! 🎉  
We welcome contributions from researchers, practitioners, students, and developers of all backgrounds. Whether you're fixing a typo, improving documentation, or adding a new biomechanical metric, your contribution is valued.

This document provides guidelines and instructions to help you get started.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Contribution Scope](#contribution-scope)
- [Making Contributions](#making-contributions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. We expect all contributors to:
- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members
- Respect data sharing ethics and privacy considerations when working with human subject data
- Provide proper attribution for contributed algorithms and methods
- Foster collaborative relationships with researchers and practitioners in biomechanics

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/stevenhirsch/force-plate-jump-analyses.git
cd force-plate-jump-analyses
```
3. Add the upstream repository:
```bash
git remote add upstream https://github.com/stevenhirsch/force-plate-jump-analyses.git
```

## Development Setup

1. Create and activate a conda environment using the provided environment files:
```bash
conda env create -f env.yml
conda activate jumpmetrics
conda env update -f local_env.yml
```

`env.yml` contains the minimal packages to build and run the package. `local_env.yml` contains additional packages for local development and batch processing.

2. Install the package in development mode:
```bash
pip install -e .
```

### Pre-commit Setup (Optional but Recommended)

To maintain code quality and consistency, we recommend setting up pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

This will automatically format code and run basic checks before commits.

## Project Structure

Understanding the project layout will help you navigate and contribute effectively:

```
force-plate-jump-analyses/
├── jumpmetrics/           # Main package code
│   ├── core/             # Core analysis functions
│   ├── events/           # Event detection
│   ├── metrics/          # Jump metric calculations
│   └── signal_processing/ # Signal processing and filtering
├── tests/                # Test suite
│   ├── core/            
│   ├── signal_processing/
│   └── example_data/    # Test data files
├── docs/                 # Documentation
├── analyses/            # Study-specific analyses
├── docker_example/      # Docker usage example
├── paper/               # Scientific paper
└── env.yml             # Environment specification
```

## Contribution Scope

There are many ways you can contribute, including:
- Fixing bugs or improving code readability
- Adding new jump metrics or biomechanical analyses
- Improving event detection or signal processing methods
- Writing or expanding documentation and tutorials
- Adding or refining tests to improve coverage and reliability
- Improving performance for large datasets
- Helping improve reproducibility (Docker, CI, etc.)

Both small and large contributions are welcome!

## Making Contributions

1. Create a new branch for your feature or bugfix.  
   Please use **lowercase with hyphens** for multiword branch names:
```bash
git checkout -b feature/new-filter-method
# or
git checkout -b fix/landing-event-bug
```

2. Make your changes, following our [Coding Standards](#coding-standards)

3. Write or update tests as needed

4. Run the test suite to ensure everything works:
```bash
python -m pytest tests/
```

5. Update documentation as necessary

## Pull Request Process

1. Update your fork with the latest upstream changes:
```bash
git fetch upstream
git merge upstream/main
```

2. Push your changes to your fork:
```bash
git push origin feature/new-filter-method
```

3. Create a Pull Request (PR) on GitHub:
   - Use a clear and descriptive title
   - Include a detailed description of the changes
   - Reference any related issues
   - Ensure all tests pass locally
   - Update documentation as needed

4. All PRs must pass **continuous integration (CI) checks** (tests, formatting, linting) before being merged.

5. Address any review comments and make necessary changes

6. Once approved, your PR will be merged

## Coding Standards

We follow PEP 8 Python style guidelines with some additional requirements:

1. **Code Style:**
   - Use 4 spaces for indentation
   - Maximum line length of 88 characters
   - Use descriptive variable and function names
   - Add docstrings for all functions, classes, and modules

2. **Type Hints:**
   - Use type hints for function arguments and return values
   - Follow PEP 484 guidelines

3. **Documentation:**
   - Write clear docstrings following NumPy style
   - Include examples in docstrings where appropriate
   - Keep inline comments minimal and meaningful

4. **Imports:**
   - Group imports in the following order:
     1. Standard library imports
     2. Third-party imports
     3. Local application imports
   - Use absolute imports when possible

5. **Scientific Computing:**
   - Prefer vectorized NumPy operations over loops for performance
   - Use appropriate units and include unit information in docstrings
   - Ensure numerical stability for biomechanical calculations

## Testing Guidelines

1. **Write tests for all new functionality**
2. **Maintain or improve code coverage**
3. **Use pytest for testing**
4. **Place tests in the appropriate subdirectory under `tests/`**
5. **Name test files with `test_` prefix**
6. **Name test functions with `test_` prefix**

### Test Data

- Place small anonymized datasets in `tests/example_data/` for reproducibility
- Do not commit sensitive or identifiable subject data
- If your test requires larger datasets, please provide mock or synthetic data

### Test Types

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test complete workflows and data processing pipelines
- **Numerical accuracy tests**: Validate calculations against known reference values
- **Performance tests**: Ensure processing speed meets requirements for large datasets

## Documentation

1. Update documentation for any new features or changes
2. Follow the existing documentation structure
3. Include docstrings for all public functions and classes
4. Update README.md if necessary
5. Add examples for new features
6. Include units and measurement context in scientific functions
7. Document data format requirements and assumptions

## Reporting Issues

When reporting issues, please help us help you by providing:

### Bug Reports
- Use the bug report template if available
- Include your operating system and Python version
- Provide a minimal reproducible example
- Include sample data (anonymized) when reporting analysis issues
- Describe expected vs. actual behavior

### Feature Requests  
- Use the feature request template if available
- Describe the scientific or practical motivation
- Provide references to relevant literature if applicable
- Consider implementation complexity and backwards compatibility

### Data-Related Issues
- When sharing example data, ensure it's anonymized
- Specify the force plate system and data format
- Include sampling rate and any preprocessing steps

## Questions or Need Help?

Feel free to:
- Open an issue for any questions
- Contact the maintainer(s) directly
- Join discussions in existing issues
- Review existing documentation and examples

We're excited to collaborate with you and look forward to your contributions! 🚀