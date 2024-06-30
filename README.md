# Samsung_OCR-Chatbot
삼성증권 과제_OCR 챗봇 서비스 개발

## 과제 설명

- 상황설명
    - 생성형 AI 기술의 발전으로 다양한 산업에 접목할 수 있는 가능성이 제시되고 있습니다. 
    금융권의 업무 프로세스에서도 이를 통해 온프레미스 형태로 사용 가능한 서비스를 개발하고자 합니다. 하지만 증권 데이터 특성상 테이블과 그래프 형태의 자료가 담긴 PDF, 이미지 파일이 많아 다양한 서비스 개발에 어려움을 겪고 있습니다. 위 문제 상황을 해결하기 위해, 여러분은 증권에서 사용할 수 있는 서비스를 제안하고, 파일 업로드 및 OCR 기능을 담아 서비스를 개발하세요
    - [ 제약 조건 ] : OpenAPI(GPT-3.5, Embedding 등), Clova API 호출 불가
    - [ 필수 포함 기능 ] : 파일 업로드, OCR,  질의 응답
- 과제 내용
    1. 개발 환경 
        - OS : Ubuntu20.04 / Cuda :12.1 / Python : 3.10
        - UI : Chainlit 혹은 streamlit
    2. 제출 파일
        1. requirements.txt
        2. Dockerfile 
        3. Docker Image 파일 혹은 docker hub 주소
        4. 결과 보고서(PPT) : 서비스 및 기능 소개, UI 시연 동영상 및 캡쳐

<br>

## Pipeline
<img width=800 src="https://github.com/namkidong98/Samsung_OCR-Chatbot/assets/113520117/8cb0dc6b-eef6-46c6-93ba-fbca5546324c">

<br><br>

## 실행 결과
<img width=500 src="https://github.com/namkidong98/Samsung_OCR-Chatbot/assets/113520117/66fe29c8-55c3-4ca7-9f8d-72e799cbb56a">
<img width=300 src="https://github.com/namkidong98/Samsung_OCR-Chatbot/assets/113520117/d1302495-b028-48b4-bacd-25ae0fe71207">

<br><br>

## 시연 영상
### PDF 파일 기반 RAG
https://github.com/namkidong98/Samsung_OCR-Chatbot/assets/113520117/7a8d697c-371b-491b-b8be-9fce3030795d

<br>

### Image 파일 기반 RAG

https://github.com/namkidong98/Samsung_OCR-Chatbot/assets/113520117/497cdcee-f351-4bd4-969e-c92994500a6e

<br><br>

## 설치 및 실행 방법

1. Git Clone
```
git clone https://github.com/namkidong98/Samsung_OCR-Chatbot.git
cd Samsung_OCR-Chatbot
```

2. 가상 환경 설치 및 활성화
```linux
conda create -n samsung python=3.10
conda activate samsung
```

3. torch 설치
```linux
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

4. 나머지 dependency 설치
```linux
pip install -r requirements.txt
```

5. embed_download.py를 실행하여 local directory에 BGE-M3 Embedding Model 설치
```
python embed_download.py
```

6. Ollama를 실행한 상태에서 llm_rag의 BASE_URL 설정
```
# BASE_URL = "http://ollama-container:11434" # Ollama-Docker를 사용한 docker-compose의 경우
# BASE_URL = "https://f9db-211-184-186-6.ngrok-free.app" # Ngrok으로 Colab 등의 GPU 서버에서 Ollama를 구동한 경우
BASE_URL = "https://localhost:11434" # Local에서 Ollama를 구동한 경우
```

7. streamlit 실행
```linux
streamlit run app.py
```

<br>

## Docker-Compose로 실행

```
docker login
docker-compose up # docker-compose.yml이 있는 폴더에서
```
<img width=800 src="https://github.com/namkidong98/Samsung_OCR-Chatbot/assets/113520117/f142cbfe-7f23-4171-877d-061d801ad2b4">
<img width=800 src="https://github.com/namkidong98/Samsung_OCR-Chatbot/assets/113520117/65288edc-add7-4b3d-ae12-398b9cca334e">
<img width=800 src="https://github.com/namkidong98/Samsung_OCR-Chatbot/assets/113520117/93bf3a5c-923a-423b-aa73-d25f7d2db6bd">
<img width=800 src="https://github.com/namkidong98/Samsung_OCR-Chatbot/assets/113520117/76f8979a-c9c2-466f-b077-014e62efbd69">

<br><br>

```yml
version: '3.8'
services:
  ollama-container:
    image: kidong98/ollama-eeve
    volumes:
      - ./data/ollama:/root/.ollama
    ports:
      - 11434:11434
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
    healthcheck:
      test: ["CMD-SHELL", "test -f /root/.ollama/models/manifests/registry.ollama.ai/library/EEVE-Korean-10.8B/latest || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 20
      start_period: 100s

  streamlit-app:
    image: kidong98/samsung-rag
    ports:
      - 8501:8501
    depends_on:
      ollama-container:
        condition: service_healthy
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
```
<img width=800 src="https://github.com/namkidong98/Samsung_OCR-Chatbot/assets/113520117/d8c5fca6-a2dd-4418-971e-97994ddb1095">

- ollama-container와 streamlit-app이라는 두 개의 컨테이너를 docker-compose를 이용하여 동시에 구동하고 관리한다
- streamlit-app은 ollama-container에서 "ollama create"가 완료되고 난 이후에 작동해야 한다
    - 이를 구현하기 위해 ollama-container의 healthcheck와 streamlit-app의 depends_on을 사용했다
    - 1. ollama-container의 healthcheck
            - ollama-container의 healthcheck에서는 100초 대기하고 30초 간격으로 test의 명령어로 체크하게 된다
            - ollama create가 성공적으로 완료되면 위 경로에 latest라는 파일이 생성되므로 이러한 파일이 있는지 체크하도록 지시한다
            - 만약 파일이 존재하지 않으면 || 뒤의 exit 1이 실행되어 healthcheck에서 unhealthy 상태가 된다
            - 만약 파일이 존재하면 스크립트가 성공적으로 완료되고 exit 0이 반환되며(정상 종료) 이를 healthcheck에서 감지하여 healthy 상태가 된다
    - 2. streamlit-app의 depends_on
            - depends_on에 있는 ollama-container가 실행된 이후에 streamlit-app이 실행될 수 있도록 한다
            - 추가적으로 condition : service_healthy로 설정하여 ollama-container가 조건을 만족하여 healthy 상태가 되기 전까지 기다리게 된다
            - 이를 통해 streamlit-app은 ollama-container에서 ollama create로 모델이 생성되기를 기다린 이후에 실행될 수 있게 된다
