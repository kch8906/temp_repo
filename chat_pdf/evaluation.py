import argparse
import numpy as np
import torch
from utils.preprocessing import (get_data_from_parquet,
                                 get_data_from_yaml,
                                 extract_corpus_col_data)
from modules import VectorDB, BM25, Rerank, Prompt, Generator
from utils.preprocessing import convert_to_nested_list

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='RAG')
parser.add_argument('--config', default=None, help='Need write a yaml file')
parser.add_argument('--corpus_data', default=None, help='')
parser.add_argument('--qa_data', default=None, help='')
args = parser.parse_args()

def main():
    config, yaml_data = get_data_from_yaml(args.config)  
    corpus_data, qa_data = get_data_from_parquet(args.corpus_data, args.qa_data)
    documents = extract_corpus_col_data(corpus_data)
    
    try:
        retrieval_array = []
        for re_gt in qa_data['retrieval_gt']:
            output_list = convert_to_nested_list(re_gt)
            retrieval_array.append(np.array(output_list))
        qa_data['retrieval_gt'] = retrieval_array
    except:
        pass
    
    queries = [query for query in qa_data['query']]
    queries = queries[:2]
   
    if 'vectordb' in config['retrieval_modules']:
        vc = VectorDB(yaml_data, documents, qa_data)
        ids_list = vc.invoke(queries)
        vc.score(ids_list, qa_data)
        del vc
        torch.cuda.empty_cache()
    if 'bm25' in config['retrieval_modules']:
        bm = BM25(yaml_data, documents, qa_data)
        ids_list = bm.invoke(corpus_data, queries)
        bm.score(ids_list, qa_data)
    
    if 'rerank' in config['node_type']:
        rr = Rerank(yaml_data, documents, qa_data, corpus_data)
        ids_list = rr.invoke(ids_list, queries)
        rr.score(ids_list, qa_data)
        del rr
        torch.cuda.empty_cache() 
    
    if 'prompt' in config['node_type']:
        pt = Prompt(yaml_data, documents, qa_data, corpus_data)
        prompt_list = pt.__dict__['prompts']
        gen = Generator(yaml_data)
        prompts = pt.invoke(ids_list, queries)
        invoke_result = gen.invoke_from_prompt(prompts)
        pt.score(invoke_result, qa_data, prompt_list)

if __name__ == "__main__":
    main()