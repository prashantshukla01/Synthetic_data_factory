import streamlit as st 
import os
from dotenv import load_dotenv 

load_dotenv()

def get_config(key_name , default = None):
    """
    Retrieves configuration with the following property:
    1. Streamlit session state(user input from ui)
    2. Environment variables
    3. Default value
    """
    
    if key_name in st.session_state and st.session_state[key_name]:
        return st.session_state[key_name]
    
    return os.getenv(key_name , default)


def set_config(key_name , value):
    """Saves a setting directly to the current session."""
    st.session_state[key_name] = value
    
def initialize_session():
    """Ensures all new platform keys exist in session state."""
    keys = [
        "TEACHER_API_KEY", "EVAL_API_KEY", "HF_TOKEN", 
        "CLOUD_PROVIDER", "CLOUD_API_KEY", "GPU_TYPE",
        "FT_TECHNIQUE", "QUANTIZATION_BITS"
    ]
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = None
            
            