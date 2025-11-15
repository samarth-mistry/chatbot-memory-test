from llm import LLM_INITIALIZED

# The "knowledge base" is now the LLM.
# So, as long as the LLM is loaded, the knowledge base is "loaded".
KNOWLEDGE_LOADED = LLM_INITIALIZED

# We no longer need the get_fact() function.
# The knowledge logic will be handled directly in the graph.