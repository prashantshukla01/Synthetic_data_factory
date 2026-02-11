import torch
import streamlit as st
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from utils.config_manager import get_config

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
    # Fetch dynamic config from UI
    model_id = get_config("BASE_MODEL_PATH", "microsoft/Phi-3-mini-4k-instruct")
    dataset_name = get_config("HF_REPO_NAME", "prashantshukla2410/synthetic-data")
    quant_mode = get_config("QUANTIZATION_MODE", "4-bit (M3/8GB Optimized)")

    st.info(f"🏗️ Preparing Training for: {model_id} on MPS...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Fix for Qwen and other models that lack a default pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # Ensure padding side is right for potential generation/training compatibility
        tokenizer.padding_side = "right"

    # Use 'mps' for Mac M3 GPU acceleration
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # FIX: On Mac M3 8GB, we use float16. 4-bit (bitsandbytes) is for NVIDIA.
    # We simulate memory efficiency using Gradient Checkpointing.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(device)

    # Enable gradient checkpointing to save RAM
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules="all-linear", 
        task_type="CAUSAL_LM"
    )

    training_args = SFTConfig(
        output_dir="./models/phi3-laptop",
        dataset_text_field="text",
        max_seq_length=512,
        per_device_train_batch_size=1, # Strict 1 for 8GB RAM
        gradient_accumulation_steps=8, # High accumulation to maintain quality
        logging_steps=1, 
        num_train_epochs=1,
        report_to="none",
        # Use standard adamw for Mac; paged_adamw is NVIDIA-only
        optim="adamw_torch" 
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=load_dataset(dataset_name, split="train"),
        peft_config=peft_config,
        args=training_args,
        tokenizer=tokenizer,
        callbacks=[StreamlitCallback(chart_placeholder)]
    )

    st.success("🚀 Training Engine Started!")
    trainer.train()
    
    # Save the adapter for the 'Adapter Library'
    trainer.save_model("./models/phi3-laptop")