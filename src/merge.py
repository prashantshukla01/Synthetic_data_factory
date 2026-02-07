from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def merge_and_save():
    # Paths based on your current project structure
    base_model_id = "microsoft/Phi-3-mini-4k-instruct"
    adapter_path = "./models/phi3-specialized-adapter" 
    save_path = "./models/genfactory-phi3-merged"

    print(" Loading base model and adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging weights into a single specialized model...")
    merged_model = model.merge_and_unload()

    print(f"Saving to {save_path}...")
    merged_model.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.save_pretrained(save_path)
    print("Done! You now have a custom specialized model folder.")

if __name__ == "__main__":
    merge_and_save()