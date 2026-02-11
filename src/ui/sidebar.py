import streamlit as st 
from utils.config_manager import set_config

def render_sidebar():
    with st.sidebar:
        st.title("Studio setting")
        
        
        # --- FEATURE 2: Dynamic Quantization ---
        
        st.subheader("Hardware and Quantisation")
        
        q_mode = st.selectbox(
            "Select Precision Mode",
            options= ["4-bit (8GB Optimized)" , "8-bit (Balanced)","Full (BF16/Cluster)"],
            index=0
        )
        
        set_config("QUANTIZATION_MODE", q_mode)
        
        hf_token = st.text_input("Hugging Face Write Token", type="password")
        set_config("HF_TOKEN", hf_token)
        
        # Old 'Model Source' section removed in favor of 'Student Model Base' below
            
        st.divider()
        
        st.info("Status: Ready to Generate")

        # --- FEATURE 4: Model Selection ---
        st.subheader("Model Configuration")
        
        st.subheader("1. Teacher (Generator)")
        
        # 1. Select Provider
        provider = st.selectbox(
            "Model Provider",
            [
                "Google Gemini", 
                "OpenAI", 
                "Anthropic", 
                "Grok (xAI)", 
                "DeepSeek", 
                "Kimi (Moonshot)"
            ],
            index=0
        )
        set_config("TEACHER_PROVIDER", provider)
        
        # 2. Dynamic Model Options & API Key
        if provider == "Google Gemini":
            model_name = st.selectbox(
                "Model Name",
                [
                    "gemini-3-pro-preview",
                    "gemini-3-flash-preview",
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                    "gemini-1.5-pro"
                ]
            )
            api_key = st.text_input("Google API Key", type="password")
            set_config("TEACHER_API_KEY", api_key)
            
            # Evaluator key is also needed for Gemini if used as judge
            eval_key = st.text_input("Evaluator API Key (Optional, defaults to above)", type="password")
            if eval_key:
                set_config("EVAL_API_KEY", eval_key)
            else:
                set_config("EVAL_API_KEY", api_key)
            
        elif provider == "OpenAI":
            model_name = st.selectbox(
                "Model Name",
                [
                    "gpt-4o", 
                    "gpt-4o-mini", 
                    "gpt-4-turbo", 
                    "gpt-3.5-turbo",
                    "o1-preview",
                    "o1-mini"
                ]
            )
            api_key = st.text_input("OpenAI API Key", type="password")
            set_config("OPENAI_API_KEY", api_key)

        elif provider == "Anthropic":
            model_name = st.selectbox(
                "Model Name",
                [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-sonnet-20240620",
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307"
                ]
            )
            api_key = st.text_input("Anthropic API Key", type="password")
            set_config("ANTHROPIC_API_KEY", api_key)
            
        elif provider == "Grok (xAI)":
            model_name = st.selectbox(
                "Model Name",
                ["grok-beta", "grok-vision-beta"]
            )
            api_key = st.text_input("xAI API Key", type="password")
            set_config("XAI_API_KEY", api_key)
            
        elif provider == "DeepSeek":
            model_name = st.selectbox(
                "Model Name",
                ["deepseek-chat", "deepseek-coder"]
            )
            api_key = st.text_input("DeepSeek API Key", type="password")
            set_config("DEEPSEEK_API_KEY", api_key)
            
        elif provider == "Kimi (Moonshot)":
            model_name = st.selectbox(
                "Model Name",
                ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]
            )
            api_key = st.text_input("Moonshot API Key", type="password")
            set_config("MOONSHOT_API_KEY", api_key)
            
        set_config("TEACHER_MODEL_NAME", model_name)
        
        # Student Model (Base for Fine-Tuning)
        student_model_type = st.radio(
            "Student Model Base",
            ["Phi-3 Mini (3.8B)", "Qwen2.5-0.5B (Tiny)", "Custom HF Repo"]
        )
        
        if student_model_type == "Phi-3 Mini (3.8B)":
            base_model_path = "microsoft/Phi-3-mini-4k-instruct"
        elif student_model_type == "Qwen2.5-0.5B (Tiny)":
            base_model_path = "Qwen/Qwen2.5-0.5B-Instruct"
        else:
            base_model_path = st.text_input("Hugging Face Repo ID", "microsoft/Phi-3-mini-4k-instruct")
            
        set_config("BASE_MODEL_PATH", base_model_path)

        