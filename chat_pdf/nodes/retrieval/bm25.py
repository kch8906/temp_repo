from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
from kiwipiepy import Kiwi, Token
from typing import List, Dict, Tuple, Callable, Union, Iterable
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from rank_bm25 import BM25Okapi
import pandas as pd
import pickle
import os


def tokenize_ko_kiwi(texts: List[str]) -> List[List[str]]:
    texts = list(map(lambda x: x.strip().lower(), texts))
    kiwi = Kiwi()
    stopwords = Stopwords()
    tokenized_list: Iterable[List[Token]] = kiwi.tokenize(texts, stopwords=stopwords)
    return [list(map(lambda x: x.form, token_list)) for token_list in tokenized_list]

BM25_TOKENIZER = {
    'ko_kiwi': tokenize_ko_kiwi,
}

def select_bm25_tokenizer(bm25_tokenizer: str) -> Callable[[str], List[Union[int, str]]]:
    if bm25_tokenizer in list(BM25_TOKENIZER.keys()):
        return BM25_TOKENIZER[bm25_tokenizer]

    return AutoTokenizer.from_pretrained(bm25_tokenizer, use_fast=False)

def get_bm25_pkl_name(bm25_tokenizer: str):
    bm25_tokenizer = bm25_tokenizer.replace('/', '')
    return f'bm25_{bm25_tokenizer}.pkl'

def load_bm25_corpus(bm25_path: str) -> Dict:
    if bm25_path is None:
        return {}
    with open(bm25_path, "rb") as f:
        bm25_corpus = pickle.load(f)
    return bm25_corpus

def validate_corpus_dataset(df: pd.DataFrame):
    columns = ['doc_id', 'contents', 'metadata']
    assert set(columns).issubset(df.columns), f"df must have columns {columns}, but got {df.columns}"

def bm25_ingest(corpus_path: str, corpus_data: pd.DataFrame, bm25_tokenizer: str = 'porter_stemmer'):
    if not corpus_path.endswith('.pkl'):
        raise ValueError(f"Corpus path {corpus_path} is not a pickle file.")
    validate_corpus_dataset(corpus_data)
    ids = corpus_data['doc_id'].tolist()

    # Initialize bm25_corpus
    bm25_corpus = pd.DataFrame()

    # Load the BM25 corpus if it exists and get the passage ids
    if os.path.exists(corpus_path) and os.path.getsize(corpus_path) > 0:
        with open(corpus_path, 'rb') as r:
            corpus = pickle.load(r)
            bm25_corpus = pd.DataFrame.from_dict(corpus)
        duplicated_passage_rows = bm25_corpus[bm25_corpus['passage_id'].isin(ids)]
        new_passage = corpus_data[~corpus_data['doc_id'].isin(duplicated_passage_rows['passage_id'])]
    else:
        new_passage = corpus_data

    if not new_passage.empty:
        tokenizer = select_bm25_tokenizer(bm25_tokenizer)
        if isinstance(tokenizer, PreTrainedTokenizerBase):
            tokenized_corpus = tokenizer(new_passage['contents'].tolist()).input_ids
        else:
            tokenized_corpus = tokenizer(new_passage['contents'].tolist())
        new_bm25_corpus = pd.DataFrame({
            'tokens': tokenized_corpus,
            'passage_id': new_passage['doc_id'].tolist(),
        })

        if not bm25_corpus.empty:
            bm25_corpus_updated = pd.concat([bm25_corpus, new_bm25_corpus], ignore_index=True)
            bm25_dict = bm25_corpus_updated.to_dict('list')
        else:
            bm25_dict = new_bm25_corpus.to_dict('list')

        bm25_dict['tokenizer_name'] = bm25_tokenizer

        with open(corpus_path, 'wb') as w:
            pickle.dump(bm25_dict, w)


def bm25_node(corpus_data: pd.DataFrame, bm25_tokenizer: str):
    bm25_dir = os.path.join("./db", 'bm25', get_bm25_pkl_name(bm25_tokenizer))
    bm25_ingest(bm25_dir, corpus_data, bm25_tokenizer=bm25_tokenizer)
    bm25_corpus = load_bm25_corpus(bm25_dir)
    bm25_instance = BM25Okapi(bm25_corpus["tokens"])
    return bm25_corpus, bm25_instance


def kiwi_tokenizer(docs):
    try:
        from kiwipiepy import Kiwi
        from kiwipiepy.utils import Stopwords
    except:
        raise AttributeError("Don't have a package, you have to 'pip install kiwipiepy'")
    stopwords = Stopwords()
    separated_words = Kiwi().tokenize(docs, stopwords=stopwords)
    # separated_words = Kiwi(model_type='sbg').tokenize(docs)
    word_list = [word.form for word in separated_words]
    return word_list