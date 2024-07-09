from typing import Dict, List
import pandas as pd
from .base import BaseModule
from nodes.rerank.korerank import (get_context_list,
                                   get_pairs_from_ids,
                                   rerank_model_load,
                                   rerank_node)
from metrics.retrieval.score import get_retrieavl_metric_to_frame

class Rerank(BaseModule):
    def __init__(self,
                 yaml_data: Dict[str, str] = None,
                 documents = None,
                 qa_data: pd.DataFrame = None,
                 corpus_data: pd.DataFrame = None):
        module_type = 'koreranker'
        super().__init__(yaml_data, module_type, documents, qa_data)
        self.corpus_data = corpus_data
        
    def invoke(self,
               ids_list: List[List[str]],
               queries: List[str]):
        context_list = get_context_list(ids_list, self.corpus_data)
        pairs = get_pairs_from_ids(ids_list, context_list, queries)  
        model, tokenizer = rerank_model_load(self.module_config['model_name'])
        ids_list = rerank_node(pairs, self.module_config['top_k'], tokenizer, model)      
        return ids_list
    
    def score(self, ids_list: List[List[str]], qa_data: pd.DataFrame):
        get_retrieavl_metric_to_frame(ids_list, qa_data, self.module_config['module_type'])
        
   
 
        
