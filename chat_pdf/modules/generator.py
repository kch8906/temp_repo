import torch
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm
from .base import BaseModule
from nodes.generator.huggingfacellm import llm_model
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DTYPE = {
    "f16": torch.float16,
    "f32": torch.float32,
    "bf16": torch.bfloat16,    
}

class Generator(BaseModule):
    def __init__(self,
                 yaml_data: Optional[Dict[str, str]] = None,):
        module_type = 'huggingface_model'
        super().__init__(yaml_data, module_type)
        self.dtype = DTYPE.get(self.module_config['dtype'], None)
        self.model, self.tokenizer = llm_model(self.module_config['model_name'], self.module_config['temperature'], self.dtype)
        
    def invoke(self):
        pass

    def invoke_from_prompt(self, prompts_: List[List[str]]) -> List[List[str]]:
        logger.info("Prompt Marker Invoke")
        invoke_result = [] 
        for idx, prompts_from_query in enumerate(prompts_):
            print(f"Prompt {idx+1} Start")
            temp_result = []
            for input in tqdm(prompts_from_query):
                outputs = self.model.generate(prompts=[input], max_new_tokens = 300, use_cache = True)
                outputs = outputs.generations[0][0].text.split("답변:")[-1]
                temp_result.append(outputs)
                torch.cuda.empty_cache()
            invoke_result.append(temp_result)
        return invoke_result
    
    def score(invoke_result: List[List[str]], qa_data: pd.DataFrame, prompt_list: List[str]):
        pass