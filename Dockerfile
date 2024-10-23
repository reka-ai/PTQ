# Start with an NVIDIA PyTorch image that includes CUDA and Python pre-installed
FROM nvcr.io/nvidia/pytorch:23.07-py3

# Set up environment variables for Python and CUDA
ENV PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda

# Install basic dependencies including OpenSSH client, Git, and required tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Configure SSH known hosts for GitHub
RUN mkdir -p /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts

# Copy the setup file and package directory into the container
COPY setup.py ./
COPY ptq ./ptq

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install the package using pip with SSH keys mounted for private repositories
RUN --mount=type=ssh python -m pip install -U pip && python -m pip install -e .

# Clean up any SSH keys after use if needed
# (Commented out because you are not copying any specific key)
# RUN rm /root/.ssh/id_rsa

# Overwrite the parent image's ENTRYPOINT
ENTRYPOINT ["python3", "ptq/examples/entrypoint_awq.py"]
