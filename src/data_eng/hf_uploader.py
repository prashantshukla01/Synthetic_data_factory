import os
from huggingface_hub import HfApi, create_repo
from utils.config_manager import get_config
from dotenv import load_dotenv

load_dotenv()

def upload_dataset_to_hf(file_path="data/processed/synthetic_data.jsonl"):
    token = get_config("HF_TOKEN")
    repo_id = get_config("HF_REPO_NAME")
    
    # CHANGES MADE HERE: Dynamic repo creation using UI token
    create_repo(repo_id=repo_id, token=token, repo_type="dataset", exist_ok=True)
    
    api = HfApi(token=token)
    api.upload_file(path_or_fileobj=file_path, path_in_repo="train.jsonl", repo_id=repo_id, repo_type="dataset")
    return f"https://huggingface.co/datasets/{repo_id}"
