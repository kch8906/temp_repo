import yaml
import functools
from typing import List, Dict

retrieval_type = ['vectordb', 'bm25', 'hybrid_rrf', 'hybrid_cc']

def preprocessing_data(func):
    @functools.wraps(func)
    def wrapper(yaml_path: str):
        yaml_data = func(yaml_path)
        node_type = [node['node_type'] for node in yaml_data['node']]
        for node in yaml_data['node']:
            if node['node_type'] == 'expansion':
                expansion_modules = [type_['module_type'] for type_ in node['module']]
            else:
                expansion_modules = None
            
            if node['node_type'] == 'retrieval':
               retrieval_modules = [type_['module_type'] for type_ in node['module']]
               break
            else:
               retrieval_modules = None
               
        config = {
            'node_type': node_type,
            'expansion_modules': expansion_modules,
            'retrieval_modules': retrieval_modules
        }
        return config, yaml_data
    return wrapper

@preprocessing_data
def get_data_from_yaml(yaml_path: str) -> Dict[str, str]:
    with open(yaml_path) as f:
        return yaml.full_load(f)

def module_parser(yaml_data: Dict[str, str], module_type: str):
    try:
        for node in yaml_data['node']:
            for type_ in node['module']:
                if type_['module_type'] == module_type:
                    return type_
    except:
        raise ValueError("The type of the retriever is not correct (string).")
    
    
def convert_to_nested_list(input_list: List[str]):
    input_str = input_list[0]
    elements = input_str.strip('[]').split(', ')
    nested_list = [[element] for element in elements]
    return nested_list  
            
            
import pandas as pd
from typing import List, Dict, Tuple
from langchain_core.documents import Document
import functools

def document_node(func) -> Document:
    @functools.wraps(func)
    def wrapper(*args):
        page_contents, metadatas = func(*args)
        if len(page_contents) == len(metadatas):
            documents = [document_format_for_courpus(page_content, metadata) for page_content, metadata in zip(page_contents, metadatas)]
        else:
            raise ValueError(f"it's different value count. page_contents({page_contents},), metadata({metadatas},)")
        return documents
    return wrapper
    
@document_node
def extract_corpus_col_data(corpus_data: pd.DataFrame) -> Tuple[List[Dict[str, str]], List[str]] :
    metadatas = [{"id":id} for id in corpus_data['doc_id']]
    page_contents =  [content for content in corpus_data['contents']]
    return page_contents, metadatas

def document_format_for_courpus(page_content: str,
                                metadata: Dict[str, str]) -> Document:
    '''
    Corpus data(DataFrame)을 Langchain Document로 변환을 위한 function
    Corpus data의 Columns는 [doc_id, contents, metadata]
    '''
    fmt = Document(
        page_content = page_content,
        metadata = metadata,
    )
    return fmt

def get_data_from_parquet(corpus_path: str,
                          qa_path: str) ->Tuple[pd.DataFrame, pd.DataFrame]:
    corpus_data = pd.read_parquet(corpus_path)
    qa_data = pd.read_parquet(qa_path)
    return corpus_data, qa_data
