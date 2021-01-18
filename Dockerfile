FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update

RUN apt-get install -y \
    wget \
    git \
    gcc \
    protobuf-compiler


# Install conda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda update -n base -c defaults conda
RUN conda install -c anaconda python=3.8

# Install Object Detection API
RUN git clone https://github.com/tensorflow/models.git
RUN cd models/research/ && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    pip3 install --use-feature=2020-resolver . && \
    cd ../.. && rm -rf models/

# Install Arial fonts
RUN apt-get install -y fontconfig
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
RUN apt-get install -y ttf-mscorefonts-installer
RUN fc-cache

# Fix "Illegal instruction (core dumped)" (on our cluster)
RUN pip install tensorflow==2.3.1 tensorflow-gpu==2.3.1

# Install cv_pipeliner
ADD requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt
RUN pip3 install label-studio>=0.7.6
RUN python3 -c 'import matplotlib.font_manager'

ADD cv_pipeliner /app/cv_pipeliner/
ADD setup.py /app/setup.py
RUN pip3 install -e /app/

# Add apps/
ADD apps /apps/apps/
WORKDIR /apps/
ENV PATH=$PATH:/apps/
ENV PYTHONPATH /apps/