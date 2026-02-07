import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

def run_smooth_test():
    # 1. Configuration for 8GB Mac M3
    repo_id = "microsoft/Phi-3-mini-4k-instruct-gguf"
    filename = "Phi-3-mini-4k-instruct-q4.gguf"
    model_path = f"./models/{filename}"

    # 2. Auto-download if missing
    if not os.path.exists(model_path):
        print(f"📥 Downloading {filename} from Hugging Face...")
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir="./models")

    # 3. Initialize with Metal (MPS) support for M3 GPU
    print("🚀 Initializing model on M3 GPU...")
    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, # Offload ALL layers to M3 GPU
            n_ctx=512,       # Keep context small for 8GB RAM
            n_threads=8,     # Utilize M3 cores
            verbose=False
        )

        test_prompt = "write 200 words essay on cricket?"
        print(f"\nUser: {test_prompt}")
        
        # 4. Generate with low memory footprint
        output = llm(
            f"<|user|>\n{test_prompt}<|end|>\n<|assistant|>",
            max_tokens=100,
            stop=["<|end|>"]
        )
        
        print(f"\n✅ Assistant: {output['choices'][0]['text'].strip()}")
        
    except Exception as e:
        print(f"❌ Error during local test: {e}")

if __name__ == "__main__":
    run_smooth_test()