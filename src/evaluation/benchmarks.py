#compares Base Model (e.g. Phi-3 vanilla) vs Fine-tuned Model.
import json
from src.evaluation.metrics import calculate_semantic_similarity

def run_regression_test(base_model , fine_tuned_model , test_dataset_path:str):
    """compares two models to ensure no performance degradation.
    """
    with open(test_dataset_path, "r") as f:
        test_data = [json.load(line) for line in f]
    
    
    results = {
        "base_avg_score": 0,
        "ft_avg_score": 0,
        "accuracy_gain": 0
    }
    
    for item in test_data:
        prompt = item['messages'][0]['content']
        expected = item['messages'][1]['content']
        
        # Get outputs from both models
        base_out = base_model.generate(prompt)
        ft_out = fine_tuned_model.generate(prompt)
        
        
        # Calculate scores
        base_score = calculate_semantic_similarity(expected, base_out)
        ft_score = calculate_semantic_similarity(expected, ft_out)
        
        
        results["base_avg_score"] += base_score
        results["ft_avg_score"] += ft_score
        
    
    # Calculate final improvement percentage
    results["base_avg_score"] /= len(test_data)
    results["ft_avg_score"] /= len(test_data)
    results["accuracy_gain"] = (results["ft_avg_score"] - results["base_avg_score"]) / results["base_avg_score"]
    
    return results