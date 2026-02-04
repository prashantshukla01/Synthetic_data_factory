import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis import RedisSaver
from src.graph.state import AgentState
from src.graph.nodes import generator_node, evaluator_node
from src.graph.edges import routing_logic
from dotenv import load_dotenv

load_dotenv()

# 1. Define your Redis URL string
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

# 2. Use 'from_conn_string' as a context manager
# This handles the connection and string parsing correctly.
with RedisSaver.from_conn_string(redis_url) as checkpointer:
    
    # 3. Setup the indices (Required for first-time use with Redis Stack)
    checkpointer.setup()

    # 4. Build and Compile the Graph
    workflow = StateGraph(AgentState)
    workflow.add_node("generator", generator_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("formatter", lambda x: x) 

    workflow.add_edge(START, "generator")
    workflow.add_edge("generator", "evaluator")
    workflow.add_conditional_edges(
        "evaluator",
        routing_logic,
        {"generator": "generator", "formatter": "formatter"}
    )
    workflow.add_edge("formatter", END)

    # 5. Compile the app within the context
    app = workflow.compile(checkpointer=checkpointer)