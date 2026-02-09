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
    api_key = (
        get_config("TEACHER_API_KEY")
        if not is_evaluator
        else get_config("EVAL_API_KEY")
    )

    model_name = "gemini-2.0-flash"

    if not api_key:
        raise ValueError("API Key is missing. Please enter it in the Sidebar.")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.7 if not is_evaluator else 0.0
    )

    return llm


def get_evaluator_llm():
    """Get or create the evaluator LLM with structured output."""
    return get_dynamic_llm(is_evaluator=True).with_structured_output(
        EvaluationSchema
    )

def generator_node(state: AgentState):
    """Generate content based on seed and feedback."""

    prompt = f"Topic: {state['seed']}\n"

    if state.get("feedback"):
        prompt += (
            f"Previous Feedback: {state['feedback']}\n"
            "Improve the output based on this feedback.\n"
        )

    llm = get_dynamic_llm(is_evaluator=False)
    response = llm.invoke(prompt)

    return {
        "current_generation": response.content,
        "messages": [response],
        "retry_count": state.get("retry_count", 0) + 1
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
        