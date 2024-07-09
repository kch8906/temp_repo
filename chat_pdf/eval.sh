#!/bin/bash

python evaluation.py --corpus_data ../data/FRM_QA_docs.parquet \
                     --qa_data ../data/FRM_QA_dataset.parquet \
	             --config ./config.yaml
