import torch
from datasets import Dataset, load_dataset
import random
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
    OpenAIGPTModel
)
import torch.nn.functional as F
from typing import Dict
import re
import os
from openai import AzureOpenAI

os.environ['HF_TOKEN'] = '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'
os.environ["AZURE_OPENAI_KEY"] = "43a2ffaecb4b4c488c992d228f2a40a3"

def get_model(model_id, is_8bit = True):
    if is_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit = True,
            bnb_8bit_use_double_quant = True,
            bnb_8bit_quant_type = "nf4",
            bnb_8bit_computer_dtype = torch.bfloat16
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    if model_id in ["UIUC-ConvAI-Sweden-GPT4", "gpt-35-turbo"]:
        return None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config = bnb_config,
            device_map = "auto",
            cache_dir = "cache"
        )
    return model

def get_tokenizer(model_id, stop_tokens=True):
    if model_id == "UIUC-ConvAI-Sweden-GPT4":
        return None
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = "cache")
    tokenizer.eos_token = "```"
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"
    
    return tokenizer

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [74694,55375,5658,14196]  # IDs of tokens where the generation should stop.
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:  # Checking if the last generated token is a stop token.
                return True
        return False

def response(model, model_id, streamer, model_inputs, temperature, device="cuda"): 
    stop = StopOnTokens()
    model_inputs = model_inputs.to(next(model.parameters()).device)
    if model_id == "UIUC-ConvAI-Sweden-GPT4":
        client = AzureOpenAI(
            azure_endpoint = "https://uiuc-convai-sweden.openai.azure.com/", 
            api_key=os.getenv("AZURE_OPENAI_KEY"),  
            api_version="2024-02-15-preview"
            )
        completion = client.chat.completions.create(
                        model="UIUC-ConvAI-Sweden-GPT4", # model = "deployment_name"
                        messages = [{"role": "user", "content": model_inputs}],
                        temperature=temperature,
                        max_tokens=200,
                        top_p=0.95,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None
                        )
        outputs = completion.choices[0].message.content
    else:
        outputs = model.generate(
            input_ids=model_inputs,
            streamer=streamer,
            max_new_tokens=200,
            early_stopping=True,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=temperature,
            repetition_penalty=1.0,
            num_beams=1,
            output_scores=True, 
            return_dict_in_generate=True,
            stopping_criteria=StoppingCriteriaList([stop]),
        )
    # print(f"\n\noutputs: \n{outputs}")
    return outputs