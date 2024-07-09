import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

def llm_model(model_name: str = "Qwen/Qwen2-7B-Instruct",
              temperature: int = 0,
              dtype: Optional[float] = torch.float16
):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config, 
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("text-generation",
                    model=base_model,
                    tokenizer=tokenizer,
                    max_new_tokens=1024,
                    model_kwargs={
                        f"temperature": {temperature},
                        "max_length": 1024},)
    hf_model = HuggingFacePipeline(pipeline=pipe)
    return hf_model, tokenizer