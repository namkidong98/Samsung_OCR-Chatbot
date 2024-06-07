# base image
FROM nvidia/cuda:12.1.0-base-ubuntu20.04

# timezone 설정 : Asia/Seoul
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 시스템 패키지 업데이트 및 Python 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create app directory
WORKDIR /app

# Copy the files
COPY requirements.txt ./
COPY app.py ./
COPY llm_rag.py ./
COPY embedding_model ./embedding_model/
COPY .streamlit ./.streamlit/
COPY images ./images/

# Install the dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]