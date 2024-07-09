import numpy as np
import pandas as pd
import functools
import evaluate
from nodes.retrieval.vectordb import LocalEmbeddingModel
from utils.util import create_Directory
from rouge import Rouge
from typing import List
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generator_metric(func):
    @functools.wraps(func)
    def wrapper(pred: List[List[str]], qa_data: pd.DataFrame) -> List[float]:
        results = []
        score_list = func(pred, qa_data)
        for case in score_list:
            results.append(sum(case)/ len(case))
        return results
    return wrapper

@generator_metric
def generator_meteor(pred: List[List[str]], qa_data: pd.DataFrame) -> List[List[float]]:
    meteor = evaluate.load('meteor')
    score_list = []
    for p in pred:
        results = [meteor.compute(predictions=[p[i]], references=[qa_data['generation_gt'][i]])['meteor'] for i in range(len(p))]
        score_list.append(results)
    return score_list

@generator_metric
def generator_rouge(pred: List[List[str]], qa_data: pd.DataFrame) -> List[List[float]]:
    rouge = Rouge()
    score_list = []
    for p in pred:
        results = [rouge.get_scores(p[i], qa_data['generation_gt'][i][0], avg=True)['rouge-l']['r'] for i in range(len(p))]
        score_list.append(results)
    return score_list

@generator_metric
def generator_semscore(pred_: List[List[str]], qa_data: pd.DataFrame) -> List[List[float]]:
    embedding_model = LocalEmbeddingModel.import_embedding_model("BAAI/bge-m3")
    score_list = []
    for p in pred_:
        answer_similar = []
        for pred, gt in zip(p, qa_data['generation_gt']):
            real_vec = embedding_model.embed_query(gt[0])
            infer_vec = embedding_model.embed_query(pred)
            answer_cosine = round(cosine_similarity(infer_vec, real_vec), 4)
            answer_similar.append(answer_cosine)
        score_list.append(answer_similar)
    return score_list


def get_generator_metric_to_frame(pred_: List[List[str]], qa_data: pd.DataFrame, prompts: List[str], module_type: str):
    logger.info("Generator Metrics")
    file_path = '../result'
    create_Directory(file_path)    
        
    meteor = generator_meteor(pred_, qa_data)
    rouge = generator_rouge(pred_, qa_data)
    semscore = generator_semscore(pred_, qa_data)
  
    df = pd.DataFrame(columns=['generator_module', 'prompt', 'generator_meteor', 'generator_rouge', 'generator_semscore'])
    for idx, score in enumerate(zip(prompts, meteor, rouge, semscore)):
        df.loc[idx] = [module_type, score[0], score[1], score[2], score[3]]
    df.to_csv(f'{file_path}/{module_type}_result.csv', index=False)
    
    # 저장도 하드코딩 없애야함 뒤에 메트릭도 해야하니 df 저장 모듈(util)
    # df.to_csv(f'{file_path}/{module_type}_result.csv', index=False)


def cosine_similarity(vec1, vec2):
    vector1 = np.array(vec1)
    vector2 = np.array(vec2)
    dot_product = np.dot(vector1, vector2)
    distance1 = np.linalg.norm(vector1, 2)
    distance2 = np.linalg.norm(vector2, 2)
    return dot_product / (distance1 * distance2)


    

    