# base img
FROM nvidia/cuda:12.1.0-base-ubuntu20.04

# timezone 설정 : Asia/Seoul
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# apt-get update
RUN apt-get update \
&& apt-get install -y wget git vim g++ gcc make curl locales \
&& rm -rf /var/lib/apt/lists/*

# miniconda 설치
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda
ENV PATH="/root/miniconda/bin:${PATH}"

# create conda env
RUN conda create -n samsung python=3.10 -y && conda init bash

# conda activate env
RUN echo "conda activate samsung" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Create app directory
WORKDIR /app

# Copy the files
COPY requirements.txt ./
COPY app.py ./
COPY llm_rag.py ./
COPY embedding_model ./embedding_model/
COPY .streamlit ./.streamlit/
COPY images ./images/

# install dependencies
RUN /root/miniconda/envs/samsung/bin/pip install -r requirements.txt

EXPOSE 8501

# Use ENTRYPOINT to run Streamlit
ENTRYPOINT ["/root/miniconda/envs/samsung/bin/streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]