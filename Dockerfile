FROM ubuntu:22.04

LABEL maintainer=" "

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
    gnupg2 \
    wget \
    python3-pip \
    python3-setuptools \
    && cd /usr/local/bin \
    && pip3 --no-cache-dir install --upgrade pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt update && apt upgrade -y
RUN apt install -y nvidia-driver-525 nvidia-dkms-525

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# RUN sudo apt-get install pciutils

# Create the /opt/ml directory
RUN mkdir -p /opt/ml
RUN mkdir -p /opt/ml/model
RUN mkdir -p /opt/ml/tmp/

COPY ./scripts/* /opt/ml

# Set the working directory to /opt/ml
WORKDIR /opt/ml
ENTRYPOINT ["/bin/bash", "-c"]

CMD ["top"]