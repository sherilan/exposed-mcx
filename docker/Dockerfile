FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Install wget, git, and rendering tools for mujoco envs
RUN apt-get update \
	&& apt-get install -y wget git libosmesa6-dev libglew2.0 libgl1-mesa-dev libglib2.0-0 unrar unzip \
	&& apt-get clean && rm -rf /var/lib/apt/lists/*
# Note, git is just for wandb to be able to grab commits
RUN git config --global --add safe.directory /workspace

# Download and extra mujoco binaries
RUN wget -qO- https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz \
	| tar -C / -xz

# Set mujoco environment
ENV MJLIB_PATH=/mujoco-2.1.1/lib/libmujoco.so.2.1.1
ENV MUJOCO_GL=egl

# Install pip requirements
COPY requirements.txt /requirements.txt
RUN pip install --ignore-installed -r /requirements.txt
RUN pip install moviepy==1.0.3
