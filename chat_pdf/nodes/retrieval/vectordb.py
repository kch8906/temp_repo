import pandas as pd
import torch
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Tuple, List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from utils.util import create_Directory, check_Vectorstore
import warnings
warnings.filterwarnings('ignore')

class LocalEmbeddingModel:
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    
    @classmethod
    def import_embedding_model(cls, model_name: str):
        return HuggingFaceEmbeddings(model_name=model_name,
                                     model_kwargs=cls.model_kwargs,
                                     encode_kwargs=cls.encode_kwargs)

def vectordb_node(documents: Document,
                  model_name: HuggingFaceEmbeddings,
                  top_k: int, queries: List[str]) -> Tuple[List[List[str]], List[List[float]]]:
    # print("[INFO] Retrieval(VectorDB)")
    embedding_model = LocalEmbeddingModel.import_embedding_model(model_name)
    save_vectorstore_to_local(documents, embedding_model)
    vectorstore = create_vectorstore(embedding_model)
    ids_list = [vectorstore.similarity_search_with_relevance_scores(query, k=top_k) for query in queries]
    
    vec_ids_list, vec_score_list = [], []
    for doc in ids_list:
        doc_list = [id_[0].metadata['id'] for id_ in doc]
        vec_list = [id_[1] for id_ in doc]
        vec_ids_list.append(doc_list)
        vec_score_list.append(vec_list)
    return vec_ids_list, vec_score_list

persist_directory: str = "./db/chroma_db"
def save_vectorstore_to_local(documents: Document, embedding_model):
    create_Directory(persist_directory)
    dir_list = check_Vectorstore(persist_directory)
    if len(dir_list) == 0:
        return Chroma.from_documents(documents, embedding_model, persist_directory=persist_directory)
    
def create_vectorstore(embedding_model):
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    
    
    