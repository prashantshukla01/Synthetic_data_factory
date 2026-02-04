from src.graph.state import AgentState

def routing_logic(state:AgentState):
    """Decides which node to visit next"""
    
    # Guardrail: Stop if we hit too many retries or a high score
    if state["score"] >=8:
        return "formatter"
    
    if state["retry_count"]>=3:
        print("Max retries reached. Forcing to formatter.")
        return "formatter"
    
    return "generator"