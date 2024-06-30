FROM ollama/ollama

# apt-get update
RUN apt-get update \
    && apt-get install -y wget curl \
    && rm -rf /var/lib/apt/lists/*

# Model 만들 준비
COPY Modelfile /app/Modelfile
WORKDIR /app
RUN wget -O ggml-model-Q5_K_M.gguf https://huggingface.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF/resolve/main/ggml-model-Q5_K_M.gguf?download=true

EXPOSE 11434