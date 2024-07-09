from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def rerank_model_load(model_name: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 'cuda'  
    model.to(device)
    return model, tokenizer

def get_context_list(ids_list: List[List[str]],
                     corpus_data: pd.DataFrame) -> List[List[str]]:
    context_list = []
    for ids in ids_list:
        contents_result = [corpus_data['contents'][corpus_data['doc_id'] == id].values[0] for id in ids]
        context_list.append(contents_result)
    return context_list

def get_pairs_from_ids(ids_list: List[List[str]],
                       context_list: List[List[str]],
                       queries: List[str]) -> List[List[Dict[str, str]]]:
    pairs = []
    for query, contents, ids in zip(queries, context_list, ids_list):
        pair = []
        for content, id in zip(contents, ids):
            result = {
                'query': query,
                'content': content,
                'id': id
            }
            pair.append(result)
        pairs.append(pair)
    return pairs

def top_n_values(arr, n: int):
    return torch.argsort(arr)[-n:].flip(dims=[0])

def rerank_node(pairs: Dict[str, str],
                top_k: int,
                tokenizer, model):
    logger.info("Retrieval Rerank")
    rerank_results = []
    for items in tqdm(pairs):
        tmp_result = []
        pairs_data = []
        for item in items:
            pairs_data.append([item['query'], item['content']])
        with torch.no_grad():
            inputs = tokenizer(pairs_data, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs_on_device = {k: v.to('cuda') for k, v in inputs.items()}
            scores = model(**inputs_on_device, return_dict=True).logits.view(-1,).float()
            scores = torch.sigmoid(scores)
        top_n = top_n_values(scores, top_k)         
        [tmp_result.append(items[idx]) for idx in top_n]
        rerank_results.append(tmp_result)
        reranked_ids = []
    for results in rerank_results:
        reranked_ids.append([results[i]['id'] for i in range(len(results))])
    return reranked_ids