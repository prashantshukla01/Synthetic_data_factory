import os 
from dotenv import load_dotenv
from pydantic import BaseModel , Field
from langchain_google_genai import ChatGoogleGenerativeAI
from src.graph.state import AgentState
load_dotenv()
class EvaluationSchema(BaseModel):
    score : int = Field(description="Score from 1 to 10", ge =1 , le= 10)
    feedback: str = Field(description="Reasoning for the score and suggestions for improvement")
    
    
    
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.7,
    api_key=os.getenv("GOOGLE_API_KEY"),
)
evaluator_llm = llm.with_structured_output(EvaluationSchema)


def generator_node(state: AgentState):
    """Generated content based on seed and feedback."""
    prompt = f"Topic:{state['seed']}\n"
    if state.get("feedback"):
        prompt += f"Previous Feedback:{state['feedback']}\n Improve the output based on this"
        
    response = llm.invoke(prompt)
    return {
        "current_generation": response.content,
        "messages":[response],
        "retry_count": state.get("retry_count", 0) +1
    }
        
        
def evaluator_node(state: AgentState):
    """Score the generated content."""
    eval_input = f"Seed: {state['seed']}\nGeneration: {state['current_generation']}"
    result = evaluator_llm.invoke(eval_input)
    
    return {
        "score": result.score,
        "feedback": result.feedback
    }
        