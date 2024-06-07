# ---------------- LLM & Embedding Model ---------------- ##
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings, HuggingFaceEmbeddings
## ---------------- VectorDB & Retriever ---------------- ##
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
## ------------------ Chain & Prompting ----------------- ##
from langchain.prompts import PromptTemplate
# from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from typing import List
from difflib import SequenceMatcher
import cv2
from easyocr import Reader
import numpy as np

USE_BGE_EMBEDDING = True # False이면 FastEmbedding
LLM_MODEL = "EEVE-Korean-10.8B:latest"
BASE_URL = "http://ollama-container:11434"
# BASE_URL = "https://fc6b-211-184-186-6.ngrok-free.app"
# BASE_URL = "https://localhost:11434"

class ChatBot:
    def __init__(self):
        # LLM 모델 설정
        self.llm = ChatOllama( 
            temperature=0,
            model=LLM_MODEL,
            base_url=BASE_URL,
            verbose=True,
        )
        self.query_llm = ChatOllama( # Query Transformation 용도
            temperature=0.2, # 기존 query로부터의 적절한 변화를 위해
            model=LLM_MODEL,
            base_url=BASE_URL,
        )
        # Embedding 모델 설정
        local_dir = "./embedding_model"
        if USE_BGE_EMBEDDING:
            # model_name = "BAAI/bge-m3"
            model_kwargs = {"device": "cuda"}
            encode_kwargs = {"normalize_embeddings": True}
            embeddings = HuggingFaceEmbeddings(
                model_name=local_dir,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        else:
            embeddings = FastEmbedEmbeddings()
        self.embeddings = embeddings
        print("Embedding Model:", self.embeddings)

        # Prompt 설정
        self.generator_prompt = PromptTemplate.from_template( # 고정
            """
            <s> [INST] 당신은 금융 및 증권 관련 전문가입니다. 제공된 맥락을 바탕으로 질문에 대해 한국어로 응답하세요.
            반드시 다음 맥락에 근거하여 응답하세요.
            적절한 근거를 찾지 못하면 '적절한 맥락을 찾지 못했습니다'라고 답하세요.
            질문, 신뢰도 등을 언급할 필요 없이 간단하고 명료하게 응답만 제공하세요. [/INST] </s> 
            [INST] 질문: {question} 
            맥락: {context} 
            응답: [/INST]
            """
        )
        self.query_prompt = PromptTemplate.from_template(
            """
            기존 질문에 대해 그 의미를 잘 나타낼 수 있도록 한 문장의 개선된 질문을 제공해주세요. 
            단, 질문에 포함된 수치는 최대한 포함해야 합니다.
            의미가 불분명한 고유 명사는 그대로 사용하세요.
            기존 질문: {question}
            수정된 질문: 
            """
        )

        # Chain 설정
        self.query_chain = self.query_prompt | self.query_llm | StrOutputParser()
        self.generator_chain = self.generator_prompt | self.llm | StrOutputParser()
        # decision_chain의 경우에는 answer 개수에 따라 달라지므로

        # Vector Store & Retriever
        self.vector_store = None
        self.retriever_sparse = None # BM25
        self.retriever_embedding = None # BGE-M3 Embedding
        # self.retriever_ensemble = None


    def pdf_ingest(self, pdf_file_path: str, pdf_file_name : str) -> None:
        docs = self.pdf2Document(filepath=pdf_file_path, file_name=pdf_file_name)  # pdf extraction
        print('-' * 25, 'Document List', '-'*25)
        for doc in docs:
            print(doc)
        print('-' * 50, '\n\n')

        # VectorStore & Retriever 설정
        self.vector_store = Chroma.from_documents(documents=docs, embedding=self.embeddings) # 임메딩 벡터 저장소 생성 및 청크 설정
        self.retriever_embedding = self.vector_store.as_retriever(
            search_type="similarity", # 유사도 스코어 기반 벡터 검색 설정
            search_kwargs={"k" :  1},
        )
        texts = [doc.page_content for doc in docs]
        metadata = [doc.metadata for doc in docs]
        self.retriever_sparse = BM25Retriever.from_texts(texts=texts, metadatas=metadata, k=1) # # BM25Retriever의 검색 결과 개수를 1로 설정
        # self.retriever_ensemble = EnsembleRetriever(
        #     retrievers=[self.retriever_dense, self.retriever_sparse], # 임베딩 기반, BM25기반(TF-IDF 기반) 순서
        #     weights=[0.5, 0.5],
        #     search_type='mmr',
        # )

    def image_ingest(self, pdf_file_path: str, pdf_file_name : str):
        reader = Reader(lang_list=['ko', 'en'], gpu=True)
        text_list, tables_list = self.image_crop_and_ocr(pdf_file_path, reader)
        docs = [] # Document로 변환하고 저장할 리스트
        
        for text in text_list: # 텍스트들은 하나로 통합해서 Document화
            text = " ".join(text.split('\n'))
            doc = Document(page_content=text, metadata={'source' : pdf_file_name, 'page' : 1})
            docs.append(doc)
        for table in tables_list: # 테이블은 하나씩 Document화
            table = " ".join(table.split('\n'))
            doc = Document(page_content=table, metadata={'source' : pdf_file_name, 'page' : 1}) 
            docs.append(doc)
        print('-' * 25, 'Document List', '-'*25)
        for doc in docs:
            print(doc)
        print('-' * 50, '\n\n')
        # VectorStore & Retriever 설정
        self.vector_store = Chroma.from_documents(documents=docs, embedding=self.embeddings) # 임메딩 벡터 저장소 생성 및 청크 설정
        self.retriever_embedding = self.vector_store.as_retriever(
            search_type="similarity", # 유사도 스코어 기반 벡터 검색 설정
            search_kwargs={"k" :  1},
        )
        texts = [doc.page_content for doc in docs]
        metadata = [doc.metadata for doc in docs]
        self.retriever_sparse = BM25Retriever.from_texts(texts=texts, metadatas=metadata, k=1) # # BM25Retriever의 검색 결과 개수를 1로 설정

    def ask(self, query: str):  # 질문 프롬프트 입력 시 호출
        if not self.vector_store:
            return "파일 업로드를 먼저 해주세요", None
        # Query Transformation
        queryset = []
        queryset.append(query)
        new_query = self.query_chain.invoke({"question": query}) # 기존 질문으로 새로운 질문 생성
        queryset.append(self.result_preprocessing(new_query)) # queryset에 추가

        # Retrieving Documents
        retrieved_docs = [] # Retrieved Documents 중 Unique만 저장
        for query in queryset:
            # docs_ensemble = self.retriever_ensemble.get_relevant_documents(query)
            docs_embedding = self.retriever_embedding.get_relevant_documents(query)
            docs_sparse = self.retriever_sparse.get_relevant_documents(query)
            print('-' * 25, "Document Retrieved", "-" * 25)
            print("Embedding Based: ", docs_embedding)
            print("BM25 Based :", docs_sparse)
            self.add_unique_document(retrieved_docs, docs_embedding[-1])
            self.add_unique_document(retrieved_docs, docs_sparse[-1])
            print('-' * 70, '\n')
        
        # Generating Answers
        answer_list = [] # 각 Document 별로 생성한 answer의 목록
        for doc in retrieved_docs: # 각 Document 별로
            answer = self.generator_chain.invoke({ # answer 생성
                "question" : query,
                "context" : doc
            })
            answer_list.append({ # 참고한 Document의 metadata와 함께 저장
                "answer" : self.result_preprocessing(answer),
                "metadata" : doc.metadata
            })

        # Final Answer selection을 위한 Prompt 구성
        prompt_content_option = ["답변1: {answer1}\n", "답변2: {answer2}\n", "답변3: {answer3}\n", "답변4: {answer4}\n"]
        decision_prompt_content = """
        당신은 금융 및 증권 관련 전문가로서 하나의 질문에 대한 답변들 중 가장 좋은 답변을 골라 말해줘야 합니다.
        가장 좋은 답변을 선택하되 선택 이유는 언급하지 말고 선택된 답변만 그대로 말해주세요.
        """
        for i in range(len(answer_list)):
            decision_prompt_content += prompt_content_option[i]
        # print(decision_prompt_content)
        invoke_option = ['answer1', 'answer2', 'answer3', 'answer4']
        invoke_basic = {"question" : queryset[0]}
        for i in range(len(answer_list)):
            invoke_basic[invoke_option[i]] = answer_list[i]['answer']
        # print(invoke_basic)
        decision_prompt = PromptTemplate.from_template(decision_prompt_content)
        decision_chain = decision_prompt | self.llm | StrOutputParser()
        final_answer = self.result_preprocessing(decision_chain.invoke(invoke_basic))
        # print(final_answer)
        final_metadata = self.find_most_similar(answer_list, final_answer)

        return final_answer, final_metadata # 응답, 메타 데이터를 같이 전송
        # return self.chain.stream( # streaming 기능
        #     {"question": query}
        # ), docs[-1].metadata # 출처 출력을 위해 metadata도 반환

    def clear(self):  # vector_store, retriever 초기화
        self.vector_store = None
        self.retriever_sparse = None
        self.retriever_dense = None
        # self.retriever_ensemble = None

    # ------------- PDF 전처리에 사용되는 함수들 -------------#
    def split_section(self, lines : List[str]) -> List[str]: # 문단 별로 구분하는 방식
        merged_lines = []
        current_line = []
        for line in lines:
            if line == '': # 두 문맥 사이에 개행이 한 번 더 있었던 경우(section을 구분할 수 있다고 판단)
                if current_line:
                    merged_lines.append(' '.join(current_line).strip())
                    current_line = []
            else:
                current_line.append(line)
        if current_line: # 마지막 줄 처리
            merged_lines.append(' '.join(current_line).strip())
        return merged_lines

    def is_table(self, line : str) -> bool: # 테이블인지 판단
        num_digits = sum(char.isdigit() for char in line) # 숫자의 개수
        total_chars = len(line) # 전체 문자의 개수
        digit_ratio = num_digits / total_chars if total_chars > 0 else 0 # 숫자의 비율
        return digit_ratio > 0.3 # 숫자의 비율이 30%를 넘으면 테이블로 판단

    def decision_for_Document(self, line : str) -> int:
        words_len = len(line.split())
        lower_bound = 30
        upper_bound = 150
        if words_len < lower_bound: return 0 # 너무 짧으면 버려라
        elif words_len < upper_bound: return 1 # 이정도 분량이면 테이블이든 텍스트든 독립적으로 추가
        else: # 이정도면 하나의 Document로 가져가기에는 분량이 길다
            if self.is_table(line): return 1 # 테이블이면, 그냥 독립적으로 추가
            else: return 2 # 텍스트이면 분할

    def pdf2Document(self, filepath, file_name) -> List[Document]:
        docs = PyPDFLoader(file_path=filepath).load()
        new_docs = []
        for doc in docs:
            page_content = doc.page_content
            page_num = doc.metadata['page'] + 1
            meta_data = {
                'page' : page_num,
                'source' : file_name,
            }
            lines = page_content.split('\n') # 개행 문자를 기준으로 분리
            lines = [line.strip() for line in lines]
            lines = self.split_section(lines)
            for line in lines:
                decision = self.decision_for_Document(line)
                if decision: # 0이면 일단 버려짐
                    if decision == 1: # 그대로 추가 
                        new_docs.append(Document(page_content=line, metadata=meta_data))
                    else: # 절반으로 나누어서 약간의 overlap을 하여 분할하기
                        boundary = len(line) // 2
                        new_docs.append(Document(page_content=line[:boundary+100], metadata=meta_data)) # overlap
                        new_docs.append(Document(page_content=line[boundary-100:], metadata=meta_data))
        return new_docs

    # ------------- ASK에 사용되는 함수들 -------------#
    def result_preprocessing(self, sentence : str) -> str: # 생성된 응답에서 불필요한 부분을 처리하는 함수
        words_to_replace = ['[INST]', '[/INST]', '수정된 질문:', '답변1:', '답변2:', '답변3:', '답변4:']
        for word in words_to_replace:
            sentence = sentence.replace(word, "")
        sentence = sentence.strip()
        return sentence

    def add_unique_document(self, doc_list: List[Document], new_doc: Document) -> None: # Document의 
        if new_doc not in doc_list:
            doc_list.append(new_doc)

    def find_most_similar(self, data : List[str], input_string : str) -> dict: # final 응답과 가장 유사한 응답의 metadata를 반환
        highest_similarity = 0
        most_similar_index = -1
        for index, item in enumerate(data):
            answer = item['answer']
            similarity = SequenceMatcher(None, answer, input_string).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_index = index
        metadata = None
        if most_similar_index != -1:
            metadata = data[most_similar_index]['metadata']
        return metadata
    
    # ------------- Image OCR에 사용되는 함수 -------------#
    def image_crop_and_ocr(self, path, reader) :
        # Load image, convert to grayscale, Otsu's threshold
        image = cv2.imread(path)
        result = image.copy()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(result, [c], -1, (36,255,12), 2)

        connect_lines = []
        table_ocr_results = []
        except_table_results = []

        for i in range(len(cnts)-1):
            for j in range(i+1, len(cnts)) :
                if ((cnts[i][0][0][0] == cnts[j][0][0][0]) and (cnts[i][1][0][0] == cnts[j][1][0][0])) :
                    connect = [cnts[i][0][0] , cnts[j][1][0]]
                    connect_lines.append(connect)
                    cv2.rectangle(result, cnts[i][0][0],  cnts[j][1][0], (0, 0, 255), -1)

                    rect = [np.array([cnts[i][0][0], cnts[j][1][0]])]
                    # 사각형 내부 이미지 추출
                    cropped_image = image[cnts[j][1][0][1]:cnts[i][0][0][1], cnts[i][0][0][0]:cnts[j][1][0][0]]
                    if (len(cropped_image) != 0 and len(cropped_image[0]) != 0) :
                        table_results = reader.readtext(cropped_image, detail=0)
                        rrr = ''
                        for r in table_results :
                            r = ''.join(r)
                            rrr = rrr+'\n'+r
                            table_ocr_results.append(rrr)
        results = reader.readtext(result, detail=0)
        rr = ''
        for r in results :
            r = ''.join(r)
            rr = rr+'\n'+r
        except_table_results.append(rr)

        ocr_results = [except_table_results, table_ocr_results]
        return ocr_results