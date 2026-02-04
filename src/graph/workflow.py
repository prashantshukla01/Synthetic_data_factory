from langgraph.graph import StateGraph , START , END
from langgraph.checkpoint.redis import RedisSaver
from src.graph.state import AgentState
from src.graph.nodes import generator_node , evaluator_node
from src.graph.edges import routing_logic
from dotenv import load_dotenv
import os

load_dotenv()

workflow = StateGraph(AgentState)
    
workflow.add_node("generator", generator_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("formatter" , lambda x:x) # Pass-through for final output

workflow.add_edge(START, "generator")
workflow.add_edge("generator", "evaluator")

workflow.add_conditional_edges(
    "evaluator",
    routing_logic,{
        "generator": "generator",
        "formatter": "formatter"
    }
)

workflow.add_edge("formatter", END)


#compile with redis persistence

redis_url = os.getenv("REDIS_URL")
checkpointer = RedisSaver.from_conn_string(redis_url)

# The 'app' object will run in main.py
app = workflow.compile(checkpointer=checkpointer)
