#
# LaSAFT Docker
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020 Loreto Parisi
#

FROM python:3.7.4-slim-buster

LABEL maintainer Loreto Parisi loretoparisi@gmail.com

WORKDIR app

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    libsndfile1 \
    ffmpeg

# app requirements
COPY src/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# app
COPY src/* ./

CMD ["bash"]