from typing import Any, Dict, List
import pandas as pd
import numpy as np
from .base import BaseModule
from nodes.retrieval.bm25 import bm25_node, kiwi_tokenizer
from utils.preprocessing import module_parser
from utils.util import create_Directory
from metrics.retrieval.score import get_retrieavl_metric_to_frame
from multiprocessing import Pool
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BM25(BaseModule):
    def __init__(self, yaml_data=None, documents=None, qa_data=None):
        module_type = 'bm25'
        super().__init__(yaml_data, module_type, documents, qa_data)
        bm25_dir = './db/bm25'
        create_Directory(bm25_dir)
        

    def invoke(self,
               corpus_data: pd.DataFrame,
               queries: List[str]):
        logger.info("Retrieval(BM25)")
        top_k = self.module_config['top_k']
        bm25_corpus, bm25_instance = bm25_node(corpus_data, self.module_config['tokenizer'])
        with Pool(processes=6) as pool:
          tokenized_queries = pool.map(kiwi_tokenizer, queries)

        ids_list = []
        bm25_score_list = []
        for query in tokenized_queries:
            scores = bm25_instance.get_scores(query)
            sorted_scores = sorted(scores, reverse=True)
            top_n_index = np.argsort(scores)[::-1][:top_k]
            ids = [bm25_corpus['passage_id'][i] for i in top_n_index]
            ids_list.append(ids)
            bm25_score_list.append(sorted_scores[:top_k])

        return ids_list
    
    def score(self, ids_list: List[List[str]], qa_data: pd.DataFrame):
        get_retrieavl_metric_to_frame(ids_list, qa_data, self.module_config['module_type'])
        
 
        
