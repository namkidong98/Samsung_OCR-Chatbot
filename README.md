# Samsung_OCR-Chatbot
삼성증권 24년 상반기 디지털/IT 프로그램, 디지털 기술팀 과제(OCR 챗봇 서비스 개발)로 대상🏆을 수상하였습니다.

## Branch 설명
1. <a href="https://github.com/namkidong98/Samsung_OCR-Chatbot">main</a> : 최종 제출 자료
2. <a href="https://github.com/namkidong98/Samsung_OCR-Chatbot/tree/docker-compose">docker-compose</a> : Docker-Compose를 이용한 구현, Ollama에서 Inference시 오류 발생

<br>

## Pipeline
<img width=1000 src="https://github.com/user-attachments/assets/ee2046aa-e093-44df-a745-d18d3b688fa1">


<br><br>

## 프로젝트 설명
### 프로젝트명
- OCR 챗봇 서비스 개발
### 프로젝트 기간
- 2024.04.12 ~ 2024.06.13 (8주)
### 프로젝트 인원
- 남기동, 이정민, 장다연, 박정주, 김대헌, 강예림
### 담당 업무
1. LLM & Embedding Model
    - On-Premise용 LLM 선정 및 설치 환경 구축
        - Ollama를 이용한 Custom LLM 생성 파이프라인 구축
        - Korean Fine-Tuning LLM으로 [**yanolja/EEVE-Korean-Instruct-10.8B-v1.0**](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0) 선정
        - 해당 모델의 Quantization 버전인 [**heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF**](https://huggingface.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF)으로 모델 생성
    - On-Premise용 Embedding Model 선정 및 설치 환경 구축
        - 임베딩 모델들(OpenAI Embedding, FastEmbedding, BGE-M3) 사이의 성능을 비교한 후 [**BAAI의 BGE-M3**](https://huggingface.co/BAAI/bge-m3)의 로컬 설치 및 활용 구축(embed_download.py)
2. RAG(Retrieval Augmented Generation)
    - Langchain을 기반으로 ChromaDB를 vectorDB로 사용하는 DPR 구조 설계
    - RAG 성능 향상을 위한 3가지 방법론 도입
        - Query Enhancement : Query LLM을 이용하여 Query Expansion을 적용(기존 query의 명료화, 다양화를 담당)
        - Retriever Enhancement : Re2G 논문을 참조하여 기존 DPR Retriever에 BM25 Retriever를 추가
        - Generator Enhancement : Re2G 논문의 Reranker 도입 아이디어를 변형하여 DPR, BM25로 검색한 Document로 응답을 생성하고 LLM에게 기존 Query에 적합한 응답을 선정하게 하는 구조(Answer-Reranking) 
3. PDF Table Extraction & Preprocessing
    - PyMuPDF, Tabular, Unstructured, PyMuPDF4LLM, Pdfplumber, PyPDF를 비교하며 증권 보고서 PDF의 테이블 데이터의 추출에 가장 적합한 라이브러리 선정(PyPDF)
    - PyPDF로 추출된 테이블 데이터를 텍스트 데이터와 분리하여 Document로 생성하기 위한 전처리 코드 작성
4. Docker
    - Docker의 Network를 이용하여 Ollama server와 Streamlit server를 동시에 실행하고 관리하는 가상화 환경 구축
    - EEVE 모델의 ollama create까지 수행하는 custom Ollama Docker Image를 생성
        - ollama/ollama를 pull한 Ollama 서버는 gguf, Modelfile을 다운로드하고 모델을 생성하는 부가적인 처리가 필요한데 이를 생략하기 위한 Custom Ollama Docker Image를 생성하고 DockerHub에 배포
        - <a href="https://hub.docker.com/r/kidong98/ollama-base-eeve">kidong98/ollama-base-eeve</a>

<br>

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

## Docker로 실행

```
docker pull kidong98/ollama-base-eeve
docker pull kidong98/samsung-llm-rag
docker network create samsung_network

docker run --gpus all -p 11434:11434 -d -e OLLAMA_HOST="0.0.0.0:11434" --name ollama-container --network samsung_network kidong98/ollama-base-eeve
docker exec -it ollama-container ollama create EEVE-Korean-10.8B -f Modelfile
# docker exec로 ollama-container 안에 모델을 생성하게 되는데 처음에 "transferring model data"가 출력되고 "success"가 출력될 때까지 약 3분 내외로 대기해줘야 한다

docker run --gpus all -p 8501:8501 --name streamlit-app --network samsung_network kidong98/samsung-llm-rag
# "success"가 출력되고 ollama-container 내의 모델 생성이 완료되면 streamlit-app을 실행하고 127.0.0.1:8501에 접속하면 된다
```

<img width=800 src="https://github.com/namkidong98/Samsung_OCR-Chatbot/assets/113520117/23c51ead-ff91-43da-b68f-848220d0eda7">

- ollama-container, streamlit-app 두 개의 서버를 하나의 Network("samsung_network")에서 구동하는 구조로 구현하였습니다
- ollama-container를 실행하는 코드에 대한 추가적인 설명
    - --gpus all : gpu 사용 옵션을 설정해줘야 Ollama에서 inference시 gpu 사용이 가능
    - --network samsung_network : streamlit-app과 동일한 네트워크 안에 실행하여 streamlit-app이 접근할 수 있도록
    - OLLAMA_HOST="0.0.0.0:11434" : 0.0.0.0으로 host를 설정해야 네트워크 내에서 streamlit-app의 접근이 허용
    - docker exec -it ollama-container ollama create: 내부의 Modelfile을 이용하여 Custom LLM을 생성할 수 있도록 명령
- streamlit-app을 실행하는 코드에 대한 추가적인 설명
    - -- gpus all : 사용자의 질문(query)에 대한 Embedding에 gpu가 사용되기 때문에 streamlit 서버에서도 gpu 옵션 사용
    - --network samsung_network : ollama-container의 LLM에 접근하기 위해 동일한 네트워크 안에 생성될 수 있도록
- 참고) 원래는 nvidia/cuda:12.1.0-base-ubuntu20.04를 기본 이미지로 하여 ollama를 curl로 설치하고 내부적으로 모델 생성하여 ollama-container를 만들고 yml 파일을 통해 docker-compose를 하였으나 이상하게 ollama에서 모델 생성 이후 loading 부분에서 오류가 발생하여 구현 방식을 수정하였습니다. 해당 부분은 같은 레포지토리의 'docker-compose' branch를 참고해주세요.

### DockerHub Link
1. ollama-container : <a href="https://hub.docker.com/r/kidong98/ollama-base-eeve">kidong98/ollama-base-eeve</a>
2. streamlit-app : <a href="https://hub.docker.com/r/kidong98/samsung-llm-rag">kidong98/samsung-llm-rag</a>

<br>

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
BASE_URL = "http://ollama-container:11434" # Docker를 사용하여 Ollama를 구동하고 Network 내에 streamlit app을 실행하는 경우
# BASE_URL = "https://f9db-211-184-186-6.ngrok-free.app" # Ngrok으로 Colab 등의 GPU 서버에서 Ollama를 구동한 경우
# BASE_URL = "https://localhost:11434" # Local에서 Ollama를 구동한 경우
```

7. streamlit 실행
```linux
streamlit run app.py
```
