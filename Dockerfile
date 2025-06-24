# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only the necessary files for package installation
COPY jumpmetrics/ /app/jumpmetrics/
COPY setup.py /app/
COPY README.md /app/
COPY LICENSE /app/

# Update pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir \
    pandas==2.0.* \
    matplotlib \
    scipy==1.14.* \
    scikit-learn==1.3.* \
    pip==23.* \
    pyyaml

# Install the package in editable mode
RUN pip install -e .

# Create directories for data mounting
RUN mkdir -p /data/input /data/output

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command to start Python interpreter
CMD ["python"] 