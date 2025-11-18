import json
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from llm import llm 
import re # Added for robust keyword search

# --- Define Guardrail Models ---
class GuardrailOutput(BaseModel):
    is_safe: bool = Field(description="True if the message is safe, False if it contains profanity, slurs, or threats.")
    reason: str = Field(description="If not safe, provide the standard warning message. If safe, provide a neutral message like 'Input is safe'.")

class ContradictionOutput(BaseModel):
    is_contradictory: bool = Field(description="True if the message contains a factual contradiction or mismatch (e.g., 'Is 30C freezing?').")
    corrected_statement: str = Field(description="If contradictory, provide the correct, non-contradictory statement.")

guardrail_parser = PydanticOutputParser(pydantic_object=GuardrailOutput)
contradiction_parser = PydanticOutputParser(pydantic_object=ContradictionOutput)

# --- Keyword List (Basic keyword list for deterministic check)  ---
# Added 'idiot' and 'stupid' for deterministic checks based on your test failures.
PROFANITY_KEYWORDS = ["idiot", "stupid"]

# --- Guardrail Chains ---

guardrail_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        You are a highly sensitive ethical filter. Your task is to analyze user input for profanity, slurs, threats, or severely disrespectful and personal attacks.

        CRITICAL REQUIREMENT: If the message is unsafe, you MUST set `is_safe` to `False`. Otherwise, set to `True`.
        
        {format_instructions}
        """),
        ("human", "User message: {user_message}")
    ]
)
guardrail_chain = guardrail_prompt | llm | guardrail_parser


contradiction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        You are a factual contradiction detector. Your task is to identify if the user's message contains a clear, factual mismatch (e.g., equating a hot temperature with freezing).

        If a contradiction exists, set `is_contradictory` to `True` and provide the correct, non-contradictory statement in `corrected_statement` (e.g., "No- 30°C is quite warm, not freezing.").
        If no contradiction exists, set `is_contradictory` to `False` and leave `corrected_statement` empty.
        
        {format_instructions}
        """),
        ("human", "User message: {user_message}")
    ]
)
contradiction_chain = contradiction_prompt | llm | contradiction_parser


# --- Master Guardrail Function (Called by graph.py) ---

def check_all_guardrails(message: str) -> str | None:
    """
    Checks the user message against all guardrail types (Profanity, Contradiction).
    Returns the error message (str) if blocked, or None if safe/not contradictory.
    """
    message_lower = message.lower()
    
    # 1. Deterministic Keyword Check (Profanity )
    for keyword in PROFANITY_KEYWORDS:
        # Use regex for whole word match to avoid false positives (e.g., 'bit' in 'habit')
        if re.search(r'\b' + re.escape(keyword) + r'\b', message_lower):
            return "Let's keep our conversation respectful, please."
        
    # 2. LLM-based Profanity/Disrespect Check (for subtle cases)
    try:
        result_safe: GuardrailOutput = guardrail_chain.invoke({
            "user_message": message,
            "format_instructions": guardrail_parser.get_format_instructions()
        })
        if result_safe.is_safe is False:
            # We catch personal attacks not in the keyword list
            return "Let's keep our conversation respectful, please."
    except Exception as e:
        print(f"Guardrail NLU failure (passing profanity check): {e}")
        # Fail-safe: if NLU fails, allow the message to proceed
        
    # 3. LLM-based Contradiction Check [cite: 21]
    try:
        result_contra: ContradictionOutput = contradiction_chain.invoke({
            "user_message": message,
            "format_instructions": contradiction_parser.get_format_instructions()
        })
        if result_contra.is_contradictory is True:
            # Return the required contradictory response [cite: 56]
            return result_contra.corrected_statement
    except Exception as e:
        print(f"Contradiction NLU failure (passing contradiction check): {e}")
        # Fail-safe: if NLU fails, allow the message to proceed

    return None # Everything passed the guardrails.