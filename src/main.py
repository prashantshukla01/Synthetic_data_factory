import uuid
import sys
from graph.workflow import app
from data_eng.formatter import format_for_finetuning, save_to_jsonl
from data_eng.hf_uploader import upload_dataset_to_hf

def run_factory(seed_list: list):
    print(f" Factory initialized with {len(seed_list)} topics.", flush=True)
    
    # Unique thread ID for Redis checkpointing
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    for i, seed in enumerate(seed_list):
        print(f"\n[{i+1}/{len(seed_list)}] 🔄 Processing Topic: {seed}...", flush=True)
        
        try:
            initial_state = {
                "seed": seed,
                "retry_count": 0,
                "messages": []
            }
            
            # This is where the AI work happens
            final_state = app.invoke(initial_state, config=config)
            
            score = final_state.get("score", 0)
            print(f"Quality Score: {score}/10", flush=True)
            
            if score >= 8:
                formatted_data = format_for_finetuning(
                    seed=final_state["seed"],
                    response=final_state["current_generation"]
                )
                save_to_jsonl(formatted_data)
                print(f"Record saved locally.", flush=True)
            else:
                print(f"Low score. Skipping save.", flush=True)
                
        except Exception as e:
            print(f"Error during graph execution: {e}", flush=True)

    print("\n Syncing all accepted data to Hugging Face...", flush=True)
    upload_dataset_to_hf()
    print(" Process Complete.", flush=True)

if __name__ == "__main__":
    # Ensure topics are defined correctly
    topics = [
        "Advanced Python Decorators",
        "C++ Memory Management",
        "Fintech API Security"
    ]
    print("--- STARTING SCRIPT ---", flush=True)
    run_factory(topics)
    
    