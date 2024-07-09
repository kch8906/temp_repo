from typing import Any, Dict, List
import pandas as pd
from .base import BaseModule
from nodes.retrieval.vectordb import vectordb_node
from utils.preprocessing import module_parser
from metrics.retrieval.score import get_retrieavl_metric_to_frame
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VectorDB(BaseModule):
    def __init__(self, yaml_data=None, documents=None, qa_data=None):
        module_type = 'vectordb'
        super().__init__(yaml_data, module_type, documents, qa_data)
        
    
    def invoke(self,
               queries: List[str]):
        logger.info("Retrieval(VectorDB)")
        ids_list, score_list = vectordb_node(self.documents, self.module_config['embedding_model'], self.module_config['top_k'], queries)
        return ids_list
    
    def score(self, ids_list: List[List[str]], qa_data: pd.DataFrame):
        get_retrieavl_metric_to_frame(ids_list, qa_data, self.module_config['module_type'])
        
 
        
