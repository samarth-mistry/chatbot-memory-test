from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import graph and module status flags
from graph import get_compiled_graph, NLU_MODULE_INITIALIZED
from knowledge import KNOWLEDGE_LOADED
# from guardrails import PROFANITY_LIST_LOADED, GIBBERISH_DETECTOR_LOADED

from graph import compiled_graph, NLU_MODULE_INITIALIZED
from knowledge import KNOWLEDGE_LOADED
from guardrails import check_all_guardrails # Renamed for clarity

# Initialize the compiled graph
compiled_graph = get_compiled_graph()

# --- App & Health Check ---

app = FastAPI()

@app.get("/healthz")
def health_check():
    """
    Health check endpoint.
    Returns 200 if all modules are loaded, 500 otherwise.
    """
    is_healthy = (
        NLU_MODULE_INITIALIZED and
        KNOWLEDGE_LOADED and
        check_all_guardrails
    )
    
    if is_healthy:
        return {"status": "ok"}
    else:
        # Be more specific about the failure
        details = {
            "nlu_module": NLU_MODULE_INITIALIZED,
            "knowledge_base": KNOWLEDGE_LOADED,
            "guardrails": check_all_guardrails
        }
        raise HTTPException(status_code=500, detail=f"Bot services not fully initialized: {details}")
    
# --- Chat Endpoint ---

class ChatRequest(BaseModel):
    user_id: str # We need this to maintain session state 
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint. [cite: 6]
    Accepts JSON and returns a text response.
    """
    user_id = request.user_id
    message = request.message

    # Config for LangGraph: specifies the "thread" or session ID
    config = {"configurable": {"thread_id": user_id}}

    # Define the initial state for a new conversation
    # We only set this if it's the *first* message
    # LangGraph's MemorySaver handles loading the state otherwise
    initial_state = {
        "user_id": user_id,
        "booking_state": "idle",
        "date_candidates": [],
    }

    # This is the input for the *current* turn
    current_input = {"message": message}

    # Run the graph
    # We use .update to add the new message
    # and .invoke to run the graph until it hits an END
    # try:
    # First, update the state with the new message and initial state (if new)
    compiled_graph.update_state(config, current_input, as_node="check_guardrails")
    
    # Now, invoke the graph to get a response
    final_state = compiled_graph.invoke(None, config)
    
    # The final response is in the 'response' key
    response_text = final_state.get('response', "I'm not sure how to respond to that.")
    
    return ChatResponse(response=response_text)
    
    # except Exception as e:
    print(f"Error during graph invocation: {e}")
    raise HTTPException(status_code=500, detail="Internal chatbot error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)