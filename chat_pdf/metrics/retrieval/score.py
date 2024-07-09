import functools
import itertools
import math
import pandas as pd
import numpy as np
from utils.util import create_Directory
from typing import List
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

metric_list = ['retrieval_f1', 'retrieval_recall', 'retrieval_precision', 'retrieval_ndcg', 'retrieval_mrr', 'retrieval_map']

def retrieval_metric(func):
    @functools.wraps(func)
    def wrapper(retrieval_gt: List[List[List[str]]], pred_ids: List[List[str]]) -> List[float]:
        results = []
        for gt, pred in zip(retrieval_gt, pred_ids):
            if gt == [[]] or any(bool(g_) is False for g in gt for g_ in g):
                results.append(None)
            else:
                results.append(func(gt, pred))
        return results

    return wrapper

@retrieval_metric
def retrieval_f1(gt: List[List[str]], pred: List[str]):
    recall_score = retrieval_recall.__wrapped__(gt, pred)
    precision_score = retrieval_precision.__wrapped__(gt, pred)
    if recall_score + precision_score == 0:
        return 0
    else:
        return 2 * (recall_score * precision_score) / (recall_score + precision_score)

@retrieval_metric
def retrieval_recall(gt: List[List[str]], pred: List[str]):
    gt_sets = [frozenset(g) for g in gt]
    pred_set = set(pred)
    hits = sum(any(pred_id in gt_set for pred_id in pred_set) for gt_set in gt_sets)
    recall = hits / len(gt) if len(gt) > 0 else 0.0
    return recall

@retrieval_metric
def retrieval_precision(gt: List[List[str]], pred: List[str]):
    gt_sets = [frozenset(g) for g in gt]
    pred_set = set(pred)
    hits = sum(any(pred_id in gt_set for gt_set in gt_sets) for pred_id in pred_set)
    precision = hits / len(pred) if len(pred) > 0 else 0.0
    return precision

@retrieval_metric
def retrieval_ndcg(gt: List[List[str]], pred: List[str]):
    gt_sets = [frozenset(g) for g in gt]
    pred_set = set(pred)
    relevance_scores = {pred_id: 1 if any(pred_id in gt_set for gt_set in gt_sets) else 0 for pred_id in pred_set}

    dcg = sum((2 ** relevance_scores[doc_id] - 1) / math.log2(i + 2) for i, doc_id in enumerate(pred))

    len_flatten_gt = len(list(itertools.chain.from_iterable(gt)))
    len_pred = len(pred)
    ideal_pred = [1] * min(len_flatten_gt, len_pred) + [0] * max(0, len_pred - len_flatten_gt)
    idcg = sum(relevance / math.log2(i + 2) for i, relevance in enumerate(ideal_pred))
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg

@retrieval_metric
def retrieval_mrr(gt: List[List[str]], pred: List[str]) -> float:
    gt_sets = [frozenset(g) for g in gt]
    rr_list = []
    for gt_set in gt_sets:
        for i, pred_id in enumerate(pred):
            if pred_id in gt_set:
                rr_list.append(1.0 / (i + 1))
                break
    return sum(rr_list) / len(gt_sets) if rr_list else 0.0

@retrieval_metric
def retrieval_map(gt: List[List[str]], pred: List[str]) -> float:
    gt_sets = [frozenset(g) for g in gt]
    ap_list = []
    for gt_set in gt_sets:
        pred_hits = [1 if pred_id in gt_set else 0 for pred_id in pred]
        precision_list = [sum(pred_hits[:i + 1]) / (i + 1) for i, hit in enumerate(pred_hits) if hit == 1]
        ap_list.append(sum(precision_list) / len(precision_list) if precision_list else 0.0)

    return sum(ap_list) / len(gt_sets) if ap_list else 0.0

def get_retrieavl_metric_to_frame(ids_list: List[List[str]], qa_data: pd.DataFrame, module_type: str):
    logger.info("Retrieval Metrics")
    file_path = '../result'
    create_Directory(file_path)    
        
    recall = retrieval_recall(qa_data['retrieval_gt'], ids_list)
    precision = retrieval_precision(qa_data['retrieval_gt'], ids_list)
    ndcg = retrieval_ndcg(qa_data['retrieval_gt'], ids_list)
    mrr = retrieval_mrr(qa_data['retrieval_gt'], ids_list)
    map = retrieval_map(qa_data['retrieval_gt'], ids_list)
    f1 = retrieval_f1(qa_data['retrieval_gt'], ids_list)
    
    df = pd.DataFrame(columns=['retrieval_module', 'retrieval_f1', 'retrieval_recall', 'retrieval_precision', 'retrieval_ndcg', 'retrieval_mrr', 'retrieval_map'])
    scores = [f1, recall, precision, ndcg, mrr, map]
    metric_score = [sum(score) / len(score) for score in scores]
    metric_score.insert(0, module_type)    

    # 0 부분 하드코딩 없애야함
    df.loc[0] = metric_score
    
    # 저장도 하드코딩 없애야함 뒤에 메트릭도 해야하니 df 저장 모듈 만들어야 할듯(util)
    df.to_csv(f'{file_path}/{module_type}_result.csv', index=False)