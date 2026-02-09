import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis import RedisSaver
from graph.state import AgentState
from graph.nodes import generator_node, evaluator_node
from graph.edges import routing_logic
from dotenv import load_dotenv

load_dotenv()


# 1. Define your Redis URL string (optional)
redis_url = os.getenv("REDIS_URL", None)

# 2. Use MemorySaver as fallback if Redis is not available
try:
    if redis_url:
        # Try to use Redis if URL is provided
        from langgraph.checkpoint.redis import RedisSaver
        with RedisSaver.from_conn_string(redis_url) as saver:
            checkpointer = saver
    else:
        # Fallback to in-memory checkpointer for local development
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
except Exception as e:
    # If Redis fails, fallback to MemorySaver
    print(f"⚠️ Redis unavailable ({e}), using MemorySaver instead")
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()


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