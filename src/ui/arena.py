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
    # Inference Logic
    user_input = st.chat_input("Ask both models a question...")
    
    if user_input:
        if not st.session_state.get("base_model") and not st.session_state.get("ft_model"):
            st.warning("⚠️ Please load at least one model first!")
        else:
            with st.chat_message("user"):
                st.write(user_input)
            
            # Inference Logic
            from ui.arena_inference import load_and_query
            from transformers import AutoTokenizer
            from utils.config_manager import get_config

            # Query Base Model
            if st.session_state.get("base_model") == "Loaded":
                with st.chat_message("assistant", avatar="🤖"):
                    status = st.empty()
                    status.write("**Base Model:** Thinking...")
                    
                    try:
                        # 1. Load Base Model
                        model = load_and_query("base")
                        
                        # 2. Get Configured Path
                        model_id = get_config("BASE_MODEL_PATH", "microsoft/Phi-3-mini-4k-instruct")
                        
                        # 3. Initialize Tokenizer
                        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                            
                        # 4. Prepare Inputs
                        # Check if chat template exists, otherwise use manual format
                        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                            messages = [{"role": "user", "content": user_input}]
                            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        else:
                            # Fallback for models without template
                            prompt = f"User: {user_input}\nAssistant:"
                            
                        inputs = tokenizer(prompt, return_tensors="pt").to("mps")
                        
                        # 5. Generate
                        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
                        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # 6. Cleanup (Naive approach, can be improved)
                        # Try to remove the prompt from the output
                        clean_response = response.replace(prompt, "").strip()
                        # Fallback cleanup
                        if "<|assistant|>" in clean_response:
                            clean_response = clean_response.split("<|assistant|>")[-1].strip()
                        
                        status.write(f"**Base Model:** {clean_response}")
                        
                    except Exception as e:
                        status.error(f"Error: {e}")

            # Query Fine-Tuned Model
            if st.session_state.get("ft_model"):
                with st.chat_message("assistant", avatar="🧠"):
                    status_ft = st.empty()
                    status_ft.write(f"**{st.session_state['ft_model']}:** Thinking...")
                    
                    try:
                        # Find adapter path
                        selected_adapter = next((a[2] for a in adapters if a[0] == st.session_state['ft_model']), None)
                        
                        if selected_adapter:
                            model = load_and_query("fine-tuned", adapter_path=selected_adapter)
                            
                            model_id = get_config("BASE_MODEL_PATH", "microsoft/Phi-3-mini-4k-instruct")
                            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                            if tokenizer.pad_token is None:
                                tokenizer.pad_token = tokenizer.eos_token
                            
                            # Same prompting logic
                            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                                messages = [{"role": "user", "content": user_input}]
                                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                            else:
                                prompt = f"User: {user_input}\nAssistant:"

                            inputs = tokenizer(prompt, return_tensors="pt").to("mps")
                            
                            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
                            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                             
                            clean_response = response.replace(prompt, "").strip()
                            if "<|assistant|>" in clean_response:
                                clean_response = clean_response.split("<|assistant|>")[-1].strip()

                            status_ft.write(f"**{st.session_state['ft_model']}:** {clean_response}")
                        else:
                            status_ft.error("Adapter path not found.")
                            
                    except Exception as e:
                        status_ft.error(f"Error: {e}")