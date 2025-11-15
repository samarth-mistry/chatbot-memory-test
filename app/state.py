from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# 1. Define the state for our graph.
# This is the "memory" of the conversation.
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        user_id: The unique ID for the user.
        message: The last message from the user.
        response: The last response from the bot.
        intent: The user's current intent (e.g., "book_reservation", "ask_fact").
        booking_state: The current status of the booking flow.
            (e.g., "idle", "awaiting_clarification", "awaiting_party_size", "complete")
        date_candidates: List of potential dates from fuzzy parsing. 
        date: The confirmed date.
        party_size: The confirmed party size.
        last_question: The question the bot just asked (to help with context).
        error_message: Any error or guardrail message to return.
    """
    user_id: str
    message: str
    response: str
    intent: str
    booking_state: str
    date_candidates: List[str]
    date: Optional[str]
    party_size: Optional[int]
    last_question: Optional[str]
    error_message: Optional[str]

# 2. Set up the in-memory session manager
# This will store a separate GraphState for each user_id. 
# This satisfies the "maintain session state for each user" requirement.
session_memory = MemorySaver()