import os 
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM , AutoTokenizer
from peft import LoraConfig
from dotenv import load_dotenv
from trl import SFTTrainer , SFTConfig

load_dotenv()

def train_laptop():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    dataset_name = "prashantshukla2410/synthetic-data"
    
    # native mac gpu check
    
    device  = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # load tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    #load model in float in float 16 (optimised for mac m3)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code = True
    ).to(device)
    
    
    #LORA setup
    
    
    
    
    

    