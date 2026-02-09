from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.config_manager import get_config
import torch
import os

def merge_and_save():
    # Paths based on your current project structure
    base_model_id = get_config("BASE_MODEL_PATH", "microsoft/Phi-3-mini-4k-instruct")
    adapter_path = "./models/phi3-laptop" # Default local path
    save_path = "./models/genfactory_final_merged"

    print(f"Merging {adapter_path} into {base_model_id}...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu"
        )
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging weights into a single specialized model...")
    merged_model = model.merge_and_unload()

    os.makedirs(save_path, exist_ok=True)
    print(f"Saving to {save_path}...")
    merged_model.save_pretrained(save_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.save_pretrained(save_path)
    
    print("Done! You now have a custom specialized model folder.")

if __name__ == "__main__":
    merge_and_save()