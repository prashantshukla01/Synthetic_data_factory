#This file ensures that the data Gemini generates is structured 
#specifically for fine-tuning. It converts the raw text into the standard {"messages": [...]} format.

import json
import os 

def format_for_finetuning(seed : str , response:str):
    """
    Converts a seed/response pair into the structured ChatML
    format used for fine-tuning small and tiny models.
    """
    
    return {
        "messages": [
            {"role": "user" , "content" : seed},
            {"role": "assistant" , "content": response}
        ]
    }
    
def save_to_jsonl(data:dict , file_path: str = "data/processed/synthetic_data.jsonl"):
    """Appends a single formatted record to a JSONL sink."""
    
    #ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")
        
    print(f"✅ Record saved to {file_path}")