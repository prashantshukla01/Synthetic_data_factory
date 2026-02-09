import streamlit as st 
from utils.config_manager import set_config

def render_sidebar():
    with st.sidebar:
        st.title("Studio setting")
        
        
        # --- FEATURE 1: Teacher & Eval API Bridge ---
        st.subheader("API Credentials")
        teacher_key = st.text_input("Larger model API Key", type="password")
        
        eval_key = st.text_input("Evaluator API Key", type = "password")
        hf_token = st.text_input("Hugging Face Write Token", type="password")
        
        # Save to session state immediately
        
        set_config("TEACHER_API_KEY", teacher_key)
        set_config("EVAL_API_KEY", eval_key)
        set_config("HF_TOKEN", hf_token)

            
        # --- FEATURE 2: Dynamic Quantization ---
        
        st.subheader("Hardware and Quantisation")
        
        q_mode = st.selectbox(
            "Select Precision Mode",
            options= ["4-bit (8GB Optimized)" , "8-bit (Balanced)","Full (BF16/Cluster)"],
            index=0
        )
        
        set_config("QUANTIZATION_MODE", q_mode)
        
        
        # --- FEATURE 3: Model Ingestion ---
        
        st.subheader("Model Source")
        model_source = st.radio("Load model from ", ["Hugging Face Hub", "Ollama (Local)","Custom Upload"])
        
        if model_source == "Hugging Face Hub":
            repo_id = st.text_input("Repo ID", "microsoft/Phi-3-mini-4k-instruct")
            set_config("BASE_MODEL_PATH", repo_id)
            
        elif model_source == "Custom Upload":
            uploaded_file = st.file_uploader("Upload Model Weights (.safetensors)")
            
        st.divider()
        
        st.info("Status: Ready to Generate")
        