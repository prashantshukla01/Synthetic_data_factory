import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from graph.state import AgentState
from utils.config_manager import get_config

load_dotenv()

class EvaluationSchema(BaseModel):
    score : int = Field(description="Score from 1 to 10", ge =1 , le= 10)
    feedback: str = Field(description="Reasoning for the score and suggestions for improvement")
    
    
    

def get_dynamic_llm(is_evaluator: bool = False):
    """
    Initializes the LLM using keys from config.
    """
    provider = get_config("TEACHER_PROVIDER", "Google Gemini")
    model_name = get_config("TEACHER_MODEL_NAME", "gemini-2.0-flash")
    
    if provider == "Google Gemini":
        api_key = get_config("TEACHER_API_KEY") if not is_evaluator else get_config("EVAL_API_KEY")
        if not api_key: raise ValueError("Google API Key is missing.")
        
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7 if not is_evaluator else 0.0
        )
        
    elif provider == "OpenAI":
        api_key = get_config("OPENAI_API_KEY")
        if not api_key: raise ValueError("OpenAI API Key is missing.")
        
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.7 if not is_evaluator else 0.0
        )
        
    elif provider == "Anthropic":
        api_key = get_config("ANTHROPIC_API_KEY")
        if not api_key: raise ValueError("Anthropic API Key is missing.")
        
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            api_key=api_key,
            temperature=0.7 if not is_evaluator else 0.0
        )
        
    elif provider == "Grok (xAI)":
        api_key = get_config("XAI_API_KEY")
        if not api_key: raise ValueError("xAI API Key is missing.")
        
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            temperature=0.7 if not is_evaluator else 0.0
        )
        
    elif provider == "DeepSeek":
        api_key = get_config("DEEPSEEK_API_KEY")
        if not api_key: raise ValueError("DeepSeek API Key is missing.")
        
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://api.deepseek.com",
            temperature=0.7 if not is_evaluator else 0.0
        )

    elif provider == "Kimi (Moonshot)":
        api_key = get_config("MOONSHOT_API_KEY")
        if not api_key: raise ValueError("Moonshot API Key is missing.")
        
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
            temperature=0.7 if not is_evaluator else 0.0
        )
    
    raise ValueError(f"Unknown provider: {provider}")


def get_evaluator_llm():
    """Get or create the evaluator LLM with structured output."""
    return get_dynamic_llm(is_evaluator=True).with_structured_output(
        EvaluationSchema
    )

def generator_node(state: AgentState):
    import time
    import random
    
    max_retries = 5
    base_delay = 10 
    
    prompt = f"Topic: {state['seed']}\n"

    if state.get("feedback"):
        prompt += (
            f"Previous Feedback: {state['feedback']}\n"
            "Improve the output based on this feedback.\n"
        )
    
    for attempt in range(max_retries):
        try:
            llm = get_dynamic_llm(is_evaluator=False)
            response = llm.invoke(prompt)

            return {
                "current_generation": response.content,
                "messages": [response],
                "retry_count": state.get("retry_count", 0) + 1
            }
            
        except Exception as e:
            # Check for Resource Exhausted (429) errors
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 5)
                print(f"⚠️ Quota exceeded. Retrying in {wait_time:.1f}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                # Re-raise other errors
                raise e
    
    # improved fallback structure
    return {
        "current_generation": "Error: Max retries reached or API Quota Exhausted.",
        "messages": [],
        "retry_count": state.get("retry_count", 0) + 1,
        "score": 0,
        "feedback": "API Error - Generation Failed" 
    }
        
        
def evaluator_node(state: AgentState):
    """Score the generated content."""
    eval_input = f"Seed: {state['seed']}\nGeneration: {state['current_generation']}"
    evaluator_llm = get_evaluator_llm()
    result = evaluator_llm.invoke(eval_input)
    
    return {
        "score": result.score,
        "feedback": result.feedback
    }
        