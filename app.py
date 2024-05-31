import os, tempfile, base64
import streamlit as st
from streamlit_chat import message
from llm_rag import ChatBot

st.set_page_config(page_title="Samsung OCR ChatBot", page_icon="🤖")

# 로컬 이미지 파일 경로
background_path = "./images/background.png"
human_icon_path = "./images/human_icon.jpg"
ai_icon_path = "./images/chat_icon.png"

# 로컬 이미지를 base64로 인코딩하는 함수
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# 배경 이미지 설정 함수
def set_background(image_path, opacity=0.3):
    bin_str = get_base64_of_bin_file(image_path)
    page_bg_img = f'''
    <style>
    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, {opacity}), rgba(255, 255, 255, {opacity})),
        url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        background-blend-mode: lighten;
    }}
    .stChatInputContainer > div {{
        background-color: rgba(255, 255, 255, {opacity}) !important;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


# 배경 이미지 설정
set_background(background_path)

def display_messages():   # 메시지 출력
    st.subheader("Chat")  
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):   # msg에는 입력/응답이 있고 is_user에는 True/False가 있다
        if is_user: # 유저의 발화
            message(msg, is_user=is_user, key=str(i))   # is_user=True인 경우 유저 입력으로 표시
        else: # AI의 발화
            message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty() # spinner 돌던 것 초기화

def process_input():   # 챗 메시지 입력 
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0: # 입력이 들어오면
        user_text = st.session_state["user_input"].strip() # user의 입력을 뽑아내서
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"): # Thinking으로 spinner 해놓고
            agent_text, metadata = st.session_state["assistant"].ask(user_text)
            # response_generator, metadata = st.session_state["assistant"].ask(user_text)
            if metadata:
                document_source = ' ['+str(metadata['source'])+' '+str(metadata['page'])+'페이지]' # 출처 정리하기
            # agent_text = st.write_stream(response_generator)   # user의 입력을 query로 하여 LLM에 보내 답변 획득
        st.session_state["messages"].append((user_text, True)) # 먼저 user 입력 넣고
        if metadata:
            st.session_state["messages"].append((agent_text + document_source, False))
        else:
            st.session_state["messages"].append((agent_text, False))

def read_and_save_file():  # file_uploader UI에서 PDF 선택 시 호출
    st.session_state["assistant"].clear()  # LLM 어시스턴스 초기화
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        file_extension = file.name.split('.')[-1].lower()
        if file_extension == 'pdf': # pdf인 경우
            with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
                st.session_state["assistant"].pdf_ingest(file_path, file.name)  # 파일을 어시스턴스에 전달
        else: # 이미지인 경우
            with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
                st.session_state["assistant"].image_ingest(file_path, file.name)  # 파일을 어시스턴스에 전달

        os.remove(file_path)

def page():
    if len(st.session_state) == 0: # 초기 설정
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatBot()  # PDF, 벡터 데이터베이스, LLM 모델 호출 역할하는 객체 설정

    # UI 정의
    st.header("Samsung OCR ChatBot")  # 타이틀  
    st.subheader("Upload a File")  # 서브헤더 
    st.file_uploader(label="Upload File",
        type=["pdf", "jpg", "jpeg", "png"], # pdf, image를 입력 받을 수 있음
        key="file_uploader", # 위젯의 상태를 참고하기 위한 키워드
        on_change=read_and_save_file, # 파일이 업로드될 때 호출될 콜백 함수
        label_visibility="collapsed",
        accept_multiple_files=True, # 여러 파일 업로드를 허용
    ) 
    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()  # 메시지 출력
    st.text_input("Message", key="user_input", on_change=process_input)  # 채팅 입력 버튼 생성

if __name__ == "__main__":
    page()