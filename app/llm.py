import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

# Load API key
load_dotenv()

# Check if the key is loaded
if not os.getenv("GOOGLE_API_KEY"):
    print("Warning: GOOGLE_API_KEY not set. LLM calls will fail.")
    LLM_INITIALIZED = False
else:
    LLM_INITIALIZED = True

# Initialize our primary LLM
# We will use this for all intelligent tasks
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- Define Pydantic Models for Structured Output ---
# This is how we force the LLM to give us structured, reliable JSON

class GuardrailOutput(BaseModel):
    """Structured output for guardrail checks."""
    is_violation: bool = Field(description="True if a violation is detected, False otherwise.")
    violation_type: str = Field(description="Type of violation (profanity, gibberish, contradiction, or none).")
    response: str = Field(description="The specific response to send to the user if a violation is detected.")

class IntentRouterOutput(BaseModel):
    """Structured output for intent routing."""
    intent: str = Field(description="The classified intent of the user.")
    
class FuzzyTimeParserOutput(BaseModel):
    """Structured output for parsing fuzzy time expressions."""
    date_candidates: List[str] = Field(description="A list of specific dates (e.g., 'saturday', 'sunday', 'monday').")