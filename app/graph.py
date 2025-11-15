import datetime # <-- Import datetime
from langgraph.graph import StateGraph, END
from state import GraphState
from llm import (
    llm, 
    IntentRouterOutput, 
    FuzzyTimeParserOutput,  # <-- NEW
    SlotSelectionOutput,    # <-- NEW
    PartySizeOutput,        # <-- NEW
    LLM_INITIALIZED
)
from guardrails import check_all_guardrails
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

NLU_MODULE_INITIALIZED = LLM_INITIALIZED

# --- Get current date for context ---
# This is CRITICAL for "fuzzy" time parsing (e.g., "this weekend")
TODAY = datetime.date.today().strftime("%A, %B %d, %Y")

# --- Define LLM-Powered Chains for Graph ---

# 1. Intent Routing Chain (Updated Prompt)
intent_parser = PydanticOutputParser(pydantic_object=IntentRouterOutput)
intent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"""
You are the central router for a chatbot. Your job is to classify the user's
intent based on their *last message* AND the *bot's last question*.
Today's date is: {TODAY}.

Possible intents are:
- 'book_reservation': User wants to start a new booking (e.g., "book a table", "I need a reservation").
- 'answer_date': User is answering the bot's question "What day would you like to book for?".
- 'answer_clarification': User is answering the bot's question about date/time (e.g., "Do you prefer [dates]?").
- 'answer_party_size': User is answering the bot's question "How many people?".
- 'ask_fact': User is asking an off-topic factual question (e.g., "capital of Australia?").
- 'confirm_booking': User is confirming the final details (e.g., "Yes, please confirm").
- 'unknown': The intent is unclear.

**Booking State Context:**
- If bot's last_question was "What day...", user's reply ("tomorrow") is 'answer_date'.
- If bot's last_question was "Do you prefer [dates]?", user's reply ("Sunday") is 'answer_clarification'.
- If bot's last_question was "How many people?", user's reply ("four") is 'answer_party_size'.
- If the user says "By the way..." or asks a question, it's 'ask_fact'.

{{format_instructions}}
"""),
        ("human", """
Bot's last question: {last_question}
User's message: {user_message}
""")
    ]
)
intent_router_chain = intent_prompt | llm | intent_parser

# 2. Factual QA Chain
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer the user's question concisely."),
        ("human", "{user_message}")
    ]
)
qa_chain = qa_prompt | llm
# --- NEW NLU CHAINS FOR PHASE 3 ---

# 3. Fuzzy Time Parser Chain
time_parser = PydanticOutputParser(pydantic_object=FuzzyTimeParserOutput)
time_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"""
You are a date/time entity extractor. Your job is to parse all fuzzy time
expressions from the user's message into a list of specific, resolved dates.
Today's date is: {TODAY}.

- 'this weekend' means {TODAY} (if Sat) and the next day, or the upcoming Saturday and Sunday.
- 'tomorrow' means the day after {TODAY}.
- 'Monday morning' just resolves to 'Monday'.
- Be specific. "this Saturday" is better than "Saturday".

{{format_instructions}}
"""),
        ("human", "User message: {user_message}")
    ]
)
fuzzy_time_parser_chain = time_prompt | llm | time_parser

# 4. Slot Selection Chain
slot_parser = PydanticOutputParser(pydantic_object=SlotSelectionOutput)
slot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are a JSON-only entity extractor. Your sole purpose is to
extract information and format it as JSON.

A user was given a list of options to choose from:
{candidates}

The user has now replied with their selection. Your job is to
identify which *exact string* from the candidate list
best matches the user's reply.

You MUST respond with a JSON object.
{format_instructions}

Do not add any other text, explanation, or conversational filler.
"""),
        ("human", "User's reply: {user_message}")
    ]
)
slot_selector_chain = slot_prompt | llm | slot_parser

# 5. Party Size Chain
party_parser = PydanticOutputParser(pydantic_object=PartySizeOutput)
party_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are a **JSON-only entity extractor**. Your single job is to
extract the party size (number of people) from the user's message
and return it *only* in the required JSON format.

You MUST NOT include any conversational text, explanation, or markdown formatting
outside of the single required JSON object.

Example: "for four" -> {{"party_size": 4}}  <--- **FIXED**

{format_instructions}
"""),
        ("human", "User's message: {user_message}")
    ]
)
party_size_chain = party_prompt | llm | party_parser


# --- Define Graph Nodes (Updated) ---

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
    # ... (unchanged from Phase 2, but uses the updated prompt) ...
    print("---NODE: route_intent_node---")
    message = state['message']
    last_question = state.get('last_question', 'None')

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

# --- REWRITTEN BOOKING NODES ---

def handle_booking_node(state: GraphState) -> GraphState:
    """
    Handles the "book_reservation" intent.
    This node parses the *initial* fuzzy time expression.
    """
    print("---NODE: handle_booking_node---")
    message = state['message']
    
    try:
        # Call the fuzzy time parser
        result: FuzzyTimeParserOutput = fuzzy_time_parser_chain.invoke({
            "user_message": message,
            "format_instructions": time_parser.get_format_instructions(),
            "today": TODAY
        })
        candidates = result.date_candidates
        
        if len(candidates) == 0:
            # No time found. Ask for it.
            state['booking_state'] = "awaiting_date"
            response = "What day would you like to book for?"
            state['response'] = response
            state['last_question'] = response
            
        elif len(candidates) == 1:
            # Time is clear. Ask for party size.
            state['date'] = candidates[0]
            state['booking_state'] = "awaiting_party_size"
            response = f"Great. How many people will be in your party on {state['date']}?"
            state['response'] = response
            state['last_question'] = response
            
        else:
            # Ambiguous time. Ask for clarification. [cite: 11]
            state['date_candidates'] = candidates
            state['booking_state'] = "awaiting_clarification"
            options = ", ".join(candidates[:-1]) + f", or {candidates[-1]}"
            response = f"Sure-just to confirm, do you mean {options}?" # 
            state['response'] = response
            state['last_question'] = response
            
    except Exception as e:
        print(f"Error in fuzzy time parsing: {e}")
        state['response'] = "I'm sorry, I had trouble understanding that time. Could you rephrase?"
        
    return state

def handle_date_node(state: GraphState) -> GraphState:
    """
    Handles the "answer_date" intent.
    The user is providing a date after we asked for it.
    """
    print("---NODE: handle_date_node---")
    message = state['message']
    
    try:
        # We can just re-use the fuzzy time parser
        result: FuzzyTimeParserOutput = fuzzy_time_parser_chain.invoke({
            "user_message": message,
            "format_instructions": time_parser.get_format_instructions(),
            "today": TODAY
        })
        candidates = result.date_candidates
        
        if len(candidates) >= 1:
            # Take the first one.
            state['date'] = candidates[0]
            state['booking_state'] = "awaiting_party_size"
            response = f"Great. How many people will be in your party on {state['date']}?"
            state['response'] = response
            state['last_question'] = response
        else:
            # Still don't get it. Re-ask.
            response = "I'm sorry, I still didn't get that. What day would you like to book for?"
            state['response'] = response
            state['last_question'] = response
            
    except Exception as e:
        print(f"Error in date NLU: {e}")
        state['response'] = "I'm sorry, I had trouble understanding that. What day?"
        
    return state

def handle_clarification_node(state: GraphState) -> GraphState:
    """
    Handles the "answer_clarification" intent.
    User is picking from the list we gave them.
    """
    print("---NODE: handle_clarification_node---")
    message = state['message']
    candidates = state['date_candidates']
    
    try:
        # Call the slot selector
        result: SlotSelectionOutput = slot_selector_chain.invoke({
            "user_message": message,
            "candidates": candidates,
            "format_instructions": slot_parser.get_format_instructions()
        })
        
        selected_date = result.selected_value
        
        if selected_date in candidates:
            # Success! Ask for party size.
            state['date'] = selected_date
            state['date_candidates'] = [] # Clear candidates
            state['booking_state'] = "awaiting_party_size"
            response = f"Great. How many people will be in your party on {state['date']}?" # 
            state['response'] = response
            state['last_question'] = response
        else:
            # LLM hallucinated or user gave a new date
            raise ValueError("Selection not in candidate list.")
            
    except Exception as e:
        print(f"Error in clarification: {e}")
        options = ", ".join(candidates[:-1]) + f", or {candidates[-1]}"
        response = f"Sorry, I didn't catch that. Do you mean {options}?"
        state['response'] = response
        
    return state

def handle_party_size_node(state: GraphState) -> GraphState:
    """
    Handles the "answer_party_size" intent.
    User is giving us the number of people.
    """
    print("---NODE: handle_party_size_node---")
    message = state['message']
    
    try:
        # Call the party size extractor
        result: PartySizeOutput = party_size_chain.invoke({
            "user_message": message,
            "format_instructions": party_parser.get_format_instructions()
        })
        
        size = result.party_size
        
        if size > 0:
            # Success! All slots filled.
            state['party_size'] = size
            state['booking_state'] = "complete" # [cite: 41]
            date = state['date']
            response = f"Perfect. Reservation for {size} on {date}. Anything else you'd like?" # 
            state['response'] = response
            state['last_question'] = "Anything else you'd like?"
        else:
            raise ValueError("Invalid party size.")

    except Exception as e:
        print(f"Error in party size extraction: {e}")
        response = "I'm sorry, I didn't catch that. How many people?"
        state['response'] = response
        state['last_question'] = "How many people?"
        
    return state
    
def handle_confirm_node(state: GraphState) -> GraphState:
    """
    Handles the "confirm_booking" intent.
    """
    print("---NODE: handle_confirm_node---")
    size = state['party_size']
    date = state['date']
    
    if size and date:
        response = f"Your table for {size} on {date} is confirmed. We'll see you then!" # 
        state['booking_state'] = "idle" # Reset for next time
        state['date'] = None
        state['party_size'] = None
    else:
        # This shouldn't happen if the router is working
        response = "I'm sorry, something went wrong. Let's start over. How can I help?"
        state['booking_state'] = "idle"
        
    state['response'] = response
    state['last_question'] = None
    return state

def handle_unknown_node(state: GraphState) -> GraphState:
    """
    Handles any message we don't understand.
    """
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
    if intent == "answer_date":           # <-- NEW
        return "handle_date"
    if intent == "answer_clarification":
        return "handle_clarification"
    if intent == "answer_party_size":
        return "handle_party_size"
    if intent == "confirm_booking":
        return "handle_confirm"
    
    return "handle_unknown"

# --- Build and Compile the Graph (Updated) ---

def get_compiled_graph():
    print("Initializing graph...")
    from state import session_memory # <-- Moved import here
    
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("check_guardrails", check_guardrails_node)
    workflow.add_node("route_intent", route_intent_node)
    workflow.add_node("handle_knowledge", handle_knowledge_node)
    workflow.add_node("handle_booking", handle_booking_node)
    workflow.add_node("handle_date", handle_date_node) # <-- NEW
    workflow.add_node("handle_clarification", handle_clarification_node)
    workflow.add_node("handle_party_size", handle_party_size_node)
    workflow.add_node("handle_confirm", handle_confirm_node)
    workflow.add_node("handle_unknown", handle_unknown_node)

    def final_response_node(state: GraphState):
        print("---NODE: final_response---")
        return state
    workflow.add_node("final_response", final_response_node)
    
    # Set the entry point
    workflow.set_entry_point("check_guardrails")

    # --- Add Edges ---
    
    workflow.add_conditional_edges(
        "check_guardrails",
        lambda state: "route_intent" if not state.get('error_message') else "final_response",
        {"route_intent": "route_intent", "final_response": "final_response"}
    )
    
    workflow.add_conditional_edges(
        "route_intent",
        route_logic,
        {
            "handle_knowledge": "handle_knowledge",
            "handle_booking": "handle_booking",
            "handle_date": "handle_date", # <-- NEW
            "handle_clarification": "handle_clarification",
            "handle_party_size": "handle_party_size",
            "handle_confirm": "handle_confirm",
            "handle_unknown": "handle_unknown"
        }
    )

    workflow.add_edge("handle_knowledge", "final_response")
    workflow.add_edge("handle_booking", "final_response")
    workflow.add_edge("handle_date", "final_response") # <-- NEW
    workflow.add_edge("handle_clarification", "final_response")
    workflow.add_edge("handle_party_size", "final_response")
    workflow.add_edge("handle_confirm", "final_response")
    workflow.add_edge("handle_unknown", "final_response")
    workflow.add_edge("final_response", END)

    # Compile the graph
    app = workflow.compile(checkpointer=session_memory)
    print("Graph compiled successfully with NLU components.")
    return app

# --- Pre-compile the graph (Updated) ---
from state import session_memory
compiled_graph = None
try:
    compiled_graph = get_compiled_graph()
    NLU_MODULE_INITIALIZED = True
except Exception as e:
    print(f"Error compiling graph: {e}")
    NSocketAddressMODULE_INITIALIZED = False