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
    peft_config = LoraConfig(
        r = 16 , 
        lora_alpha = 32,
        target_modules = "all-linear" , task_type = "CAUSAL_LM"
    )
    
    training_args = SFTConfig(
        output_dir = "./models/phi3-laptop",
        dataset_text_field= "text",
        max_seq_length = 512,
        num_train_epochs =1,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size = 1,
        save_strategy="no",
        report_to = "none",
        )
    
    trainer = SFTTrainer(
        model = model,
        train_dataset = load_dataset(dataset_name , split="train"),
        peft_config = peft_config,
        args = training_args,
        tokenizer = tokenizer
        
    )
    trainer.train()
    
if __name__ == "__main__":
    train_laptop()
    
    
    
    

    