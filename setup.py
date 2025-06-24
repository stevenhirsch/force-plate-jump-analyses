import os
import re
import yaml
from setuptools import setup, find_packages

# Get the directory where setup.py is located
here = os.path.abspath(os.path.dirname(__file__))

# Get version from __init__.py
with open(os.path.join(here, "jumpmetrics", "__init__.py"), "r") as f:
    init_content = f.read()
    version_match = re.search(r'__version__ = ["\']([^"\']*)["\']', init_content)
    if version_match:
        package_version = version_match.group(1)
        print(f"Extracted version: {package_version}")
    else:
        raise RuntimeError("Unable to find version string.")

# Try to read dependencies from env.yml
try:
    with open(os.path.join(here, "env.yml"), "r") as f:
        env_yaml = yaml.safe_load(f)
        
    # Extract dependencies - typically under 'dependencies' key in conda env files
    conda_dependencies = env_yaml.get('dependencies', [])

    # Filter out non-pip installable packages and extract pip section
    pip_dependencies = []

    for dep in conda_dependencies:
        if isinstance(dep, dict) and 'pip' in dep:
            # Extract pip dependencies and clean wildcards
            for pip_dep in dep['pip']:
                # Handle wildcards by removing them (e.g., pandas>=2.0.* -> pandas>=2.0)
                if '*' in pip_dep:
                    pip_dep = pip_dep.split('*')[0].rstrip('.')
                pip_dependencies.append(pip_dep)
        elif isinstance(dep, str):
            # Skip python version spec
            if dep.startswith('python'):
                continue
            
            # Convert conda dependency format to pip format
            # Handle different formats like 'package=1.0.0' or 'package==1.0.0'
            dep = dep.replace('==', '=')  # Normalize separators
            
            # Handle wildcards by removing them
            if '*' in dep:
                dep = dep.split('*')[0].rstrip('.')
                
            parts = dep.split('=')
            package = parts[0]
            
            if len(parts) > 1:
                dep_version = parts[1].rstrip('.')  # Remove trailing dots that might be before a wildcard
                pip_dependencies.append(f"{package}>={dep_version}")
            else:
                pip_dependencies.append(package)
except FileNotFoundError:
    # Fallback to basic dependencies if env.yml is not found
    print("Warning: env.yml not found, using default dependencies.")
    pip_dependencies = [
        "build"
    ]

# Try to read readme
try:
    with open(os.path.join(here, "readme.md"), "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A package for computing jump metrics from force plate data."

print(f"Final version being used: {package_version}")
setup(
    name="jumpmetrics",
    version=package_version,
    author="Steven Hirsch",
    author_email="stevehirsch94@gmail.com",
    description="A package for computing jump metrics from force plate data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stevenhirsch/force-plate-jump-analyses",
    packages=find_packages(
        exclude=["tests", "tests.*", "analyses", "analyses.*", "docker_example", "docker_example.*"]
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Biomechanics",
    ],
    python_requires=">=3.10",
    install_requires=pip_dependencies,
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
)