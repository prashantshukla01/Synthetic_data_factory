import uuid
from src.graph.workflow import app
from src.data_eng.formatter import format_for_finetuning, save_to_jsonl
from src.data_eng.hf_uploader import upload_dataset_to_hf


def run_factory(seed_list : list):
    """
    Runs the AI factory for a list of topics.
    """
    
    #unique thread id for redis checkpointing
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    
    for seed in seed_list:
        print(f"\n--- Processing Topic: {seed} ---")
        
        
        #run the graph
        
        initial_state = {
            "seed": seed,
            "retry_count": 0,
            "messages": []
        }
        final_state = app.invoke(initial_state , config=config)
        
        #extract and format if score is high enough
        
        if final_state.get("score" , 0) >= 8:
            formatted_data = format_for_finetuning(
                seed = final_state["seed"],
                response = final_state["current_generation"]
                )
        
        
        #save to local jsonl
            save_to_jsonl(formatted_data)
            
        else:
            print(f"Topic '{seed}' failed to meet quality score")
            
    upload_dataset_to_hf()
    
    if __name__ == "__main__":
        #example seeds
        seeds = [
            "The impact of climate change on global agriculture",
            "Advancements in renewable energy technologies",
            "The role of artificial intelligence in modern healthcare"
        ]
        
        run_factory(seeds)