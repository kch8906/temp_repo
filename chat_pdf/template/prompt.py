from llama_index.core import PromptTemplate

def qa_prompt_tmpl(func):
    prompt = str(func)
    prompt_tmpl = PromptTemplate(prompt)
    return prompt_tmpl

def prompt_1():
    prompt = """
사용자 질문과 관련된 구절이 있습니다. 다음 질문에 신중하게 답변해 주세요.
Answer the question. Think step by step.

Passage: {context_str}

Question: {query_str}

answer:
"""
    return PromptTemplate(prompt)