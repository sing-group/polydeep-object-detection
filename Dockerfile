FROM mxnet/python:1.4.1_gpu_cu92_mkl_py3
LABEL maintainer="anrodriguez"

# INSTALL COMPI
ADD image-files/compi.tar.gz /

# PLACE HERE YOUR DEPENDENCIES (SOFTWARE NEEDED BY YOUR PIPELINE)
RUN apt-get update && apt-get install gettext curl openjdk-9-jre-headless python3-tk gnuplot ffmpeg bc -y && apt-get autoremove --purge && apt-get autoclean
RUN apt-get install -y jq
RUN pip install numpy==1.14.6
RUN pip install gluoncv==0.7.0
RUN pip install portalocker==1.5.1

# Download and configure miniconda3
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN rm -rf /root/miniconda3
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH=/root/miniconda3/bin:$PATH
RUN conda update -n base -c defaults conda

# Create the environment:
COPY environment_cuda.yml .
RUN conda env create -f environment_cuda.yml

ADD ttv.xml /ttv.xml
ADD train.xml /train.xml
ADD test.xml /test.xml
ADD video-annotation.xml /video-annotation.xml
ADD scripts /scripts
WORKDIR /
