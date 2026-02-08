import torch
import streamlit as st
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# 1. Custom Callback for Streamlit
class StreamlitCallback(TrainerCallback):
    def __init__(self, chart_placeholder):
        self.chart_placeholder = chart_placeholder
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])
            # Update the Streamlit chart live
            self.chart_placeholder.line_chart(self.losses)

def train_laptop(chart_placeholder):
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    dataset_name = "prashantshukla2410/synthetic-data"
    
    # Force Redownload: Delete local cache reference if needed
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 4-bit Quantization Config for 8GB RAM Optimization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load Model with Quantization
    # Note: device_map="auto" is often required for quantization, 
    # but for MPS we might need to be careful. 
    # Current bitsandbytes + accelerate on Mac might prefer explicit device or auto.
    # We will try low_cpu_mem_usage=True and let accelerate handle map if possible, 
    # or manual to(device) if bnb/mps allows.
    # For now, keeping manual to(device) removed if we use quantization as it conflicts often.
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    # model.to(device) # Quantized models often don't support .to() directly or are already mapped.

    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"
    )

    training_args = SFTConfig(
        output_dir="./models/phi3-laptop",
        dataset_text_field="text",
        max_seq_length=512,
        per_device_train_batch_size=1, # Reduced for 8GB RAM
        gradient_accumulation_steps=8, # Increased to maintain effective batch size
        gradient_checkpointing=True,   # Critical for memory saving
        logging_steps=1, # Log every step for smooth UI updates
        num_train_epochs=1,
        report_to="none", # Disable wandb to prevent UI freezing
        optim="paged_adamw_8bit" # Optimize optimizer memory
    )

    # Initialize Trainer with the Callback
    trainer = SFTTrainer(
        model=model,
        train_dataset=load_dataset(dataset_name, split="train"),
        peft_config=peft_config,
        args=training_args,
        tokenizer=tokenizer,
        callbacks=[StreamlitCallback(chart_placeholder)]
    )

    trainer.train()