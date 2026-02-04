from typing_extensions import TypedDict , Annotated , List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # The raw "seed" or instruction provided at the start
    seed : str
    # The current generated draft from the Generator node
    current_generation :str
    
    score: int 
    # The reasoning/feedback from the evaluator on how to improve
    feedback: str
    # Counter to prevent infinite loops in the Retry path
    retry_count: int
    
    