FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update

RUN apt-get install -y \
    wget \
    git \
    gcc \
    protobuf-compiler

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda update -n base -c defaults conda
RUN conda install -c anaconda python=3.8

# Install Object Detection API
ADD https://api.github.com/repos/tensorflow/models/git/refs/heads/master version.json
RUN rm version.json && git clone https://github.com/tensorflow/models.git
RUN cd models/research/ && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    pip3 install --use-feature=2020-resolver . && \
    cd ../.. && rm -rf models/

# Install two-stage-pipeliner
ADD two_stage_pipeliner /tmp/two_stage_pipeliner/
ADD setup.py /tmp/setup.py
ADD requirements.txt /tmp/requirements.txt
RUN pip3 install --use-feature=2020-resolver /tmp/ && \
    rm -rf /tmp/two_stage_pipeliner/ && \
    rm -rf /tmp/setup.py && \
    rm -rf /tmp/requirements.txt

ADD app_main.py /app/app_main.py
ADD app_config.yaml /app/app_config.yaml
WORKDIR /app/
CMD ["streamlit", "run", "app_main.py", "--server.port", "80"]
