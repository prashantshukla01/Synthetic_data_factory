import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer , SFTConfig


def train_supercomputer():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    
    #4 bit quantization nf4 (normal float 4)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16  # bf16 is best for NVIDIA
    )
    
    # Load Model with Auto-sharding
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" # Automatically handles multi-GPU
    )
    
    # Training Args (Scalable for Supercomputer)
    
    training_args = SFTConfig(
        output_dir="./models/phi3-cluster",
        bf16=True, # High-performance NVIDIA training
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        logging_steps=1,
        report_to="tensorboard" # Or 'wandb' for college graphs
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=load_dataset("prashantshukla2410/synthetic-data", split="train"),
        args=training_args
    )
    trainer.train()
    
if __name__ == "__main__":
    train_supercomputer()
    
    