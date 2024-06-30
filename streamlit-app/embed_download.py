# Embedding Model Local Directory에 미리 저장
from sentence_transformers import SentenceTransformer
local_dir = "./embedding_model" # 로컬 저장 경로
model_name = 'BAAI/bge-m3'  # 사용할 모델 이름
model = SentenceTransformer(model_name)
model.save('./embedding_model')