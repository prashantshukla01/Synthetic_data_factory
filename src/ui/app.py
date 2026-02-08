import streamlit as st
from src.training.trainer_laptop import train_laptop

st.title("🏭 GenFactory Production Dashboard")

if st.button("🚀 Start Laptop Fine-Tuning"):
    st.subheader("Live Training Metrics")
    
    # Create placeholders for the live graph
    chart_placeholder = st.empty()
    status_text = st.empty()
    
    status_text.info("Downloading model and starting MPS engines...")
    
    # Run training and pass the UI placeholder
    train_laptop(chart_placeholder)
    
    status_text.success("✅ Training Complete! Model saved to ./models/phi3-laptop")