import streamlit as st
from utils.config_manager import initialize_session
from ui.sidebar import render_sidebar
from graph.workflow import app

from training.trainer_laptop import train_laptop
from ui.arena import render_arena


st.set_page_config(page_title="Universal Studio",layout= "wide")
initialize_session()

render_sidebar()

st.title("GenFactory: Universal Fine-Tuning Studio")

tab1 , tab2 , tab3 = st.tabs(["Production", "Dataset Scrubber", "Comparison Arena"])

with tab1:
    st.header("Generate Synthetic Data")
    topic = st.text_input("Enter Topic Seed", "Advanced Python Concurrency")
    
    if st.button("Generate and Eval"):
        # Fix: State expects 'seed', not 'topic'
        result = app.invoke({"seed": topic, "messages": []}) 
        st.write(result)
        
    st.divider()
    st.subheader("Fine-Tuning (Laptop/MPS)")
    if st.button("Start Training"):
        chart_placeholder = st.empty()
        train_laptop(chart_placeholder)
        
with tab2:
    st.header("Step 2: Human-in-the-Loop Curation")
    st.info("Edit your generated dataset below before starting the fine-tune.")
    

with tab3:
    
    st.header("Step 3: The Arena")
    render_arena()