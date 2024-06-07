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
<img width=800 src="https://github.com/namkidong98/Samsung_OCR-Chatbot/assets/113520117/a1b28006-2f16-4f35-b2ce-99c9cc86d139">

<br><br>

## 실행 결과
<img width=500 src="https://github.com/namkidong98/Samsung_OCR-Chatbot/assets/113520117/66fe29c8-55c3-4ca7-9f8d-72e799cbb56a">
<img width=300 src="https://github.com/namkidong98/Samsung_OCR-Chatbot/assets/113520117/d1302495-b028-48b4-bacd-25ae0fe71207">

<br><br>

## 시연 영상
https://github.com/namkidong98/Samsung_OCR-Chatbot/assets/113520117/7a8d697c-371b-491b-b8be-9fce3030795d

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




