FROM ubuntu:20.04

# python
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-venv \
    sudo \
    pep8 \
    git \
    vim

# For click python package
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install Python dependencies
COPY requirements.txt /app/
RUN python3.8 -m pip install -r /app/requirements.txt

COPY app/* /app/app/
COPY server.py /app/

WORKDIR /app
# Disabling stdout buffering to see the logs instantly
ENV PYTHONUNBUFFERED=1
CMD ["stdbuf", "-o0", "python3", "server.py"]

