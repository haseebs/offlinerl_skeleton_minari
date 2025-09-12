# Use a Python base image
# FROM python:3.9-slim
FROM riotshield/mujoco210:built AS base

FROM base

# Set the environment variable globally
# ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/root/.mujoco/mujoco210/:$LD_LIBRARY_PATH

ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
# Append to .bashrc for shell sessions
# RUN echo 'export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/root/.mujoco/mujoco210/:$LD_LIBRARY_PATH' >> /root/.bashrc


# Set the working directory in the container
WORKDIR /app

# Install git and other dependencies
RUN apt-get update && apt-get install -y git && apt-get clean

# Copy the project files into the container
COPY . /app

# # Install Python dependencies
# COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# install mysql
RUN apt-get update && apt-get install -y mysql-client && rm -rf /var/lib/apt/lists/*


# Default command (can be overridden by Docker Compose)
CMD ["python", "main.py" "run=0"]
# ENTRYPOINT ["python"]
