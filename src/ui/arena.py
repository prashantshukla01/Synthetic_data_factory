import streamlit as st
from database.adapter_lib import list_adapters
# New Import for inference
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def render_arena():
    st.subheader("⚔️ The Comparison Arena")
    
    # CHANGES MADE HERE: Pulling dynamic adapter list from your DB
    adapters = list_adapters()
    adapter_names = [a[0] for a in adapters]
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.info("Model A: Base (Vanilla)")
        if st.button("Load Base Model"):
            # Load in float16 for M3 safety
            st.session_state["base_model"] = "Loaded" 
            st.success("Base Model Ready")
        
    with col_b:
        selected = st.selectbox("Model B: Fine-Tuned", adapter_names if adapter_names else ["No Adapters Found"])
        if st.button("Load Fine-Tuned"):
            st.session_state["ft_model"] = selected
            st.success(f"{selected} Ready")

    st.divider()
    user_input = st.chat_input("Ask both models a question...")
    # Inference logic will go here next