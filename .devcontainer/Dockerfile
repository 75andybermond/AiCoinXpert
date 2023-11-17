FROM mcr.microsoft.com/devcontainers/base:debian
#ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7
ARG DEBIAN_FRONTEND=noninteractive
ARG USER=vscode
RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update \ 
    && apt-get install -y build-essential --no-install-recommends make \
        ca-certificates \
        git \
        libglib2.0-0 \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        xz-utils \
        libsqlite3-dev \
        tk-dev \
        libgl1-mesa-glx \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        v4l-utils \
        && usermod -aG video $USER

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Python and poetry installation
USER $USER
ARG HOME="/home/$USER"
ARG PYTHON_VERSION=3.11
# ARG PYTHON_VERSION=3.11
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${HOME}/.local/bin:$PATH"
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
RUN echo "done 0" \
    && curl https://pyenv.run | bash \
    && echo "done 1" \
    && pyenv install ${PYTHON_VERSION} \
    && echo "done 2" \
    && pyenv global ${PYTHON_VERSION} \
    && echo "done 3" \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.in-project true
# Add LD_LIBRARY_PATH environment variable
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# ENV NVIDIA_REQUIRE_CUDA "cuda>=8.0"
