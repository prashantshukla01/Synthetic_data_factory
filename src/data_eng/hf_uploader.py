import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

def upload_dataset_to_hf(local_path: str = "data/processed/synthetic_data.jsonl"):
    """
    Uploads the synthetic data to prashantshukla2410/synthetic-data.
    """
    api = HfApi()
    token = os.getenv("HF_TOKEN")
    repo_id = "prashantshukla2410/synthetic-data" 
    
    if not token:
        print("❌ Error: HF_TOKEN not found in .env. Please add your Write Token.")
        return

    try:
        print(f"🚀 Pushing data to https://huggingface.co/datasets/{repo_id}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo="train.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        print("🎉 Success! Your dataset is now updated on Hugging Face.")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    # You can run this file standalone to test the upload
    upload_dataset_to_hf()