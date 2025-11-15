from langgraph.graph import StateGraph, END
from state import GraphState
from llm import llm, IntentRouterOutput, LLM_INITIALIZED
from guardrails import check_llm_guardrails
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# --- Health Check Flag ---
NLU_MODULE_INITIALIZED = LLM_INITIALIZED

# --- Define LLM-Powered Chains for Graph ---

# 1. Intent Routing Chain
intent_parser = PydanticOutputParser(pydantic_object=IntentRouterOutput)
intent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are the central router for a chatbot. Your job is to classify the user's
intent based on their *last message* AND the *bot's last question*.

Possible intents are:
- 'book_reservation': User wants to start a new booking.
- 'answer_clarification': User is answering the bot's question about date/time.
- 'answer_party_size': User is answering the bot's question about party size.
- 'ask_fact': User is asking an off-topic factual question (e.g., "capital of Australia?").
- 'confirm_booking': User is confirming the final details.
- 'unknown': The intent is unclear.

**Booking State Context:**
- If bot's last_question was "Do you prefer [dates]?", user's reply ("Sunday") is 'answer_clarification'.
- If bot's last_question was "How many people?", user's reply ("four") is 'answer_party_size'.
- If the user says "By the way..." or asks a question, it's 'ask_fact'.
- If the user says "book a table" or "I want a reservation", it's 'book_reservation'.

{format_instructions}
"""),
        ("human", """
Bot's last question: {last_question}
User's message: {user_message}
""")
    ]
)
intent_router_chain = intent_prompt | llm | intent_parser

# 2. Factual QA Chain (simpler, no Pydantic needed)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer the user's question concisely."),
        ("human", "{user_message}")
    ]
)
qa_chain = qa_prompt | llm

# --- Define Graph Nodes (Updated) ---

def check_guardrails_node(state: GraphState) -> GraphState:
    """
    First node: Check for adversarial or unethical input.
    """
    print("---NODE: check_guardrails_node---")
    # We now call our new LLM-powered guardrail
    error_msg = check_llm_guardrails(state['message'])
    if error_msg:
        state['response'] = error_msg
        state['error_message'] = error_msg # Signal to end
    return state

def route_intent_node(state: GraphState) -> GraphState:
    """
    Second node: The main "router".
    Uses an LLM to decide where to go next.
    """
    print("---NODE: route_intent_node---")
    message = state['message']
    last_question = state.get('last_question', 'None')

    # Call the LLM router
    try:
        result: IntentRouterOutput = intent_router_chain.invoke({
            "last_question": last_question,
            "user_message": message,
            "format_instructions": intent_parser.get_format_instructions()
        })
        print(f"---LLM Router classified intent as: {result.intent}---")
        state['intent'] = result.intent
    except Exception as e:
        print(f"Error in intent routing: {e}")
        state['intent'] = "unknown"
        
    return state

def handle_knowledge_node(state: GraphState) -> GraphState:
    """
    Handles a factual query by calling the LLM.
    This replaces the 20-fact JSON.
    """
    print("---NODE: handle_knowledge_node---")
    message = state.get('message', '')
    
    # Call the LLM QA chain
    try:
        llm_response = qa_chain.invoke({"user_message": message})
        fact = llm_response.content
    except Exception as e:
        print(f"Error in QA chain: {e}")
        fact = "I'm sorry, I'm having trouble looking that up right now."
    
    # Use .get with a safe default to avoid KeyError
    booking_state = state.get('booking_state', "idle")
    last_question = state.get('last_question', "")
    
    # This is the "resume" logic
    if booking_state not in [None, "idle", "complete"]:
        state['response'] = f"{fact} Now, back to your reservation... {last_question}"
    else:
        state['response'] = fact
        
    state['intent'] = None # Clear intent
    return state

# --- STUB NODES FOR PHASE 3 ---
# We still need to build the NLU logic, but the router is ready
# to send us here.

def handle_booking_node(state: GraphState) -> GraphState:
    print("---NODE: handle_booking_node---")
    # This node will be upgraded in Phase 3 to use the LLM
    # to parse "this weekend or maybe Monday morning".
    
    # For now, we'll stub the clarification question.
    state['date_candidates'] = ["saturday", "sunday", "monday"]
    state['booking_state'] = "awaiting_clarification"
    response = "Sure-just to confirm, do you mean Saturday, Sunday, or Monday morning?"
    state['response'] = response
    state['last_question'] = response
    
    return state

def handle_clarification_node(state: GraphState) -> GraphState:
    print("---NODE: handle_clarification_node---")
    # In Phase 3, we'll parse the user's answer ("Sunday")
    state['date'] = "Sunday" # Stub
    state['booking_state'] = "awaiting_party_size"
    response = "Great. How many people will be in your party on Sunday?"
    state['response'] = response
    state['last_question'] = response
    return state

def handle_party_size_node(state: GraphState) -> GraphState:
    print("---NODE: handle_party_size_node---")
    # In Phase 3, we'll parse "four"
    state['party_size'] = 4 # Stub
    state['booking_state'] = "complete"
    response = "Perfect. Reservation for four on Sunday. Anything else you'd like?"
    state['response'] = response
    state['last_question'] = "Anything else you'd like?"
    return state
    
def handle_confirm_node(state: GraphState) -> GraphState:
    print("---NODE: handle_confirm_node---")
    state['response'] = "Your table for four this Sunday is confirmed. We'll see you then!"
    state['booking_state'] = "idle" # Reset for next time
    state['last_question'] = None
    return state

def handle_unknown_node(state: GraphState) -> GraphState:
    print("---NODE: handle_unknown_node---")
    state['response'] = "I'm sorry, I don't understand. Can you rephrase?"
    return state

# --- Define Graph Edges (Updated) ---

def route_logic(state: GraphState) -> str:
    """
    Conditional edge: The main router logic.
    Reads the 'intent' set by the LLM.
    """
    intent = state['intent']
    
    if intent == "ask_fact":
        return "handle_knowledge"
    if intent == "book_reservation":
        return "handle_booking"
    if intent == "answer_clarification":
        return "handle_clarification"
    if intent == "answer_party_size":
        return "handle_party_size"
    if intent == "confirm_booking":
        return "handle_confirm"
    
    return "handle_unknown"

# --- Build and Compile the Graph ---

def get_compiled_graph():
    print("Initializing graph...")
    
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("check_guardrails", check_guardrails_node)
    workflow.add_node("route_intent", route_intent_node)
    workflow.add_node("handle_knowledge", handle_knowledge_node)
    workflow.add_node("handle_booking", handle_booking_node)
    workflow.add_node("handle_clarification", handle_clarification_node)
    workflow.add_node("handle_party_size", handle_party_size_node)
    workflow.add_node("handle_confirm", handle_confirm_node)
    workflow.add_node("handle_unknown", handle_unknown_node)

    # Final response node (no change)
    def final_response_node(state: GraphState):
        print("---NODE: final_response---")
        return state
    workflow.add_node("final_response", final_response_node)
    
    # Set the entry point
    workflow.set_entry_point("check_guardrails")

    # --- Add Edges ---
    
    # 1. Guardrail edge
    workflow.add_conditional_edges(
        "check_guardrails",
        lambda state: "route_intent" if not state.get('error_message') else "final_response",
        {"route_intent": "route_intent", "final_response": "final_response"}
    )
    
    # 2. Main router edge
    workflow.add_conditional_edges(
        "route_intent",
        route_logic, # Use our new LLM-powered router function
        {
            "handle_knowledge": "handle_knowledge",
            "handle_booking": "handle_booking",
            "handle_clarification": "handle_clarification",
            "handle_party_size": "handle_party_size",
            "handle_confirm": "handle_confirm",
            "handle_unknown": "handle_unknown"
        }
    )

    # 3. Edges to final response
    workflow.add_edge("handle_knowledge", "final_response")
    workflow.add_edge("handle_booking", "final_response")
    workflow.add_edge("handle_clarification", "final_response")
    workflow.add_edge("handle_party_size", "final_response")
    workflow.add_edge("handle_confirm", "final_response")
    workflow.add_edge("handle_unknown", "final_response")
    
    # The end node
    workflow.add_edge("final_response", END)

    # Compile the graph
    app = workflow.compile(checkpointer=session_memory)
    print("Graph compiled successfully with LLM components.")
    return app

# --- Pre-compile the graph (No change) ---
from state import session_memory
compiled_graph = None
try:
    compiled_graph = get_compiled_graph()
    NLU_MODULE_INITIALIZED = True
except Exception as e:
    print(f"Error compiling graph: {e}")
    NLU_MODULE_INITIALIZED = False