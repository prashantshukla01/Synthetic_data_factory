# two-layer evaluation and hallucination-detection mechanism for generated model outputs

import torch 
from sentence_transformers import SentenceTransformer , util
from src.graph.nodes import llm # reusing gemini for evaluation


# Load a lightweight embedding model for similarity checks
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_semantic_similarity(reference:str , candidate:str):
    """Measures how close the student model is to the 'Gold Standard'."""
    emb1 = similarity_model.encode( reference , convert_to_tensor=True )
    emb2 = similarity_model.encode( candidate , convert_to_tensor=True )
    return util.pytorch_cos_sim(emb1 , emb2).item()

def check_hallucination(input_text : str , generated_text: str):
    """Uses Gemini as a 'Judge' to detect factual errors."""
    prompt = f""" Compare the following AI response to the source input.
    Input: {input_text}
    Response: {generated_text}
    
    Task: Identify any factually incorrect information or 'hallucinations'. 
    Return a score from 1 (Totally Hallucinated) to 10 (Perfectly Grounded).
    """
    
    response = llm.invoke(prompt)
    # Extract numeric score from judge output
    return response.content