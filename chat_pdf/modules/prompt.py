from typing import Any, Dict, List
import pandas as pd
from langchain.prompts import PromptTemplate
from .base import BaseModule
from transformers import AutoTokenizer
from metrics.generator.score import get_generator_metric_to_frame

class Prompt(BaseModule):
    def __init__(self, yaml_data=None, documents=None, qa_data=None, corpus_data=None):
        module_type = 'prompt_marker'
        super().__init__(yaml_data, module_type, documents, qa_data)
        self.corpus_data = corpus_data
        # self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", add_bos_token=True, trust_remote_code=True)
        self.prompts = self.module_config['prompt_list']
        
    def invoke(self,
               ids_list: List[List[str]],
               queries: List[str]):
        prompts = []
        
        for template in self.prompts:
            input_list = []
            for idx, ids in (enumerate(ids_list)):
                contents = [self.corpus_data['contents'][self.corpus_data['doc_id'] == id_].values[0] for id_ in ids]
                # inputs = self.tokenizer(
                #     [
                #         template.format(
                #             contents,
                #             queries[idx],
                #         )
                #     ], return_tensors = "pt").to("cuda")

                fmt_prompt = template.format(
                    contents=contents,
                    question=queries[idx],)
                input_list.append(fmt_prompt)
            prompts.append(input_list)
        return prompts
    
    def score(self, invoke_result: List[List[str]], qa_data: pd.DataFrame, prompt_list: List[str]):
        get_generator_metric_to_frame(invoke_result, qa_data, prompt_list, self.module_config['module_type'])
        
   
 
        
