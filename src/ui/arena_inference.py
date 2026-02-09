import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils.config_manager import get_config

def clear_vram():
    """Forces the Mac to release memory between model swaps."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

def load_and_query(model_type, adapter_path=None):
    """
    Loads one model at a time to stay under 8GB.
    model_type: 'base' or 'fine-tuned'
    """
    clear_vram() # Step 1: Clear everything before loading
    
    model_id = get_config("BASE_MODEL_PATH", "microsoft/Phi-3-mini-4k-instruct")
    
    # Step 2: Load Model in 16-bit for M3 stability
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="mps"
    )
    
    # Step 3: If testing fine-tuned, attach the LoRA
    if model_type == 'fine-tuned' and adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        
    return model # In a real chat, you would run .generate() here