from langgraph.graph import StateGraph, END
from state import GraphState, session_memory
from guardrails import check_all_guardrails
from knowledge import get_fact

# --- Define Graph Nodes ---
# Nodes are functions that modify the state

def check_guardrails_node(state: GraphState) -> GraphState:
    """
    First node: Check for adversarial or unethical input.
    """
    print("---NODE: check_guardrails_node---")
    error_msg = check_all_guardrails(state['message'])
    if error_msg:
        state['response'] = error_msg
        state['error_message'] = error_msg # Signal to end
    return state

def route_intent_node(state: GraphState) -> GraphState:
    """
    Second node: The main "router".
    Decides where to go next based on user intent and current state.
    """
    print("---NODE: route_intent_node---")
    message = state['message'].lower()

    # --- This is the Topic Switch logic [cite: 13] ---
    fact = get_fact(message)
    if fact:
        state['intent'] = "ask_fact"
        return state
    # --- End Topic Switch ---

    # Simple NLU stub
    if "book" in message or "reservation" in message:
        state['intent'] = "book_reservation"
    elif state['booking_state'] == "awaiting_clarification":
        state['intent'] = "provide_clarification"
    else:
        state['intent'] = "unknown"
        
    return state

def handle_knowledge_node(state: GraphState) -> GraphState:
    """
    Handles a factual query.
    """
    print("---NODE: handle_knowledge_node---")
    message = state['message'].lower()
    fact = get_fact(message)
    
    # This is the "resume" logic 
    # We check if we were in the middle of something.
    if state['booking_state'] != "idle":
        state['response'] = f"{fact} Now, back to your reservation... {state['last_question']}"
    else:
        state['response'] = fact
        
    state['intent'] = None # Clear intent
    return state

def handle_booking_node(state: GraphState) -> GraphState:
    """
    Handles the booking intent.
    This is where we'll add NLU for fuzzy time  in Phase 3.
    """
    print("---NODE: handle_booking_node---")
    
    # STUB for Phase 1
    # In Phase 3, we'll parse "weekend or Monday" here.
    state['response'] = "Sure, I can help with that. What day?"
    state['last_question'] = "What day?"
    state['booking_state'] = "awaiting_party_size" # Stub
    
    return state

def handle_unknown_node(state: GraphState) -> GraphState:
    """
    Handles any message we don't understand.
    """
    print("---NODE: handle_unknown_node---")
    state['response'] = "I'm sorry, I don't understand. Can you rephrase?"
    return state

# --- Define Graph Edges ---
# Edges decide which node to run next.

def should_continue(state: GraphState) -> str:
    """
    Conditional edge: Called after guardrails.
    If 'error_message' is set, we end the graph.
    """
    if state.get('error_message'):
        return "end_graph"
    return "route_intent"

def route_logic(state: GraphState) -> str:
    """
    Conditional edge: The main router logic.
    Reads the 'intent' set by the router node.
    """
    if state['intent'] == "ask_fact":
        return "handle_knowledge"
    if state['intent'] == "book_reservation":
        return "handle_booking"
    
    # We'll add more routes here in Phase 3
    # e.g., "provide_clarification" -> "handle_slot_filling_node"
    
    return "handle_unknown"

# --- Build and Compile the Graph ---

def get_compiled_graph():
    """
    Builds and compiles the LangGraph state machine.
    """
    # This satisfies the "NLU module initialized" for /healthz 
    print("Initializing graph...")
    
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("check_guardrails", check_guardrails_node)
    workflow.add_node("route_intent", route_intent_node)
    workflow.add_node("handle_knowledge", handle_knowledge_node)
    workflow.add_node("handle_booking", handle_booking_node)
    workflow.add_node("handle_unknown", handle_unknown_node)

    # Add a special node that just returns the final response
    def final_response_node(state: GraphState) -> GraphState:
        # This node doesn't do anything, it's just a clean
        # exit point for the graph to return the 'response'.
        print("---NODE: final_response---")
        return state

    workflow.add_node("final_response", final_response_node)
    
    # Set the entry point
    workflow.set_entry_point("check_guardrails")

    # Add edges
    workflow.add_conditional_edges(
        "check_guardrails",
        should_continue,
        {
            "route_intent": "route_intent",
            "end_graph": "final_response" # Go to end if guardrail hit
        }
    )
    
    workflow.add_conditional_edges(
        "route_intent",
        route_logic,
        {
            "handle_knowledge": "handle_knowledge",
            "handle_booking": "handle_booking",
            "handle_unknown": "handle_unknown"
        }
    )

    # After these nodes run, go to the end
    workflow.add_edge("handle_knowledge", "final_response")
    workflow.add_edge("handle_booking", "final_response")
    workflow.add_edge("handle_unknown", "final_response")
    
    # The end node
    workflow.add_edge("final_response", END)

    # Compile the graph
    app = workflow.compile(checkpointer=session_memory)
    print("Graph compiled successfully.")
    return app

# --- Pre-compile the graph on startup ---
NLU_MODULE_INITIALIZED = False
try:
    compiled_graph = get_compiled_graph()
    NLU_MODULE_INITIALIZED = True
except Exception as e:
    print(f"Error compiling graph: {e}")