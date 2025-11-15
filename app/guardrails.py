from llm import llm, GuardrailOutput
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from typing import Optional

# This flag now checks if the LLM itself is ready
from llm import LLM_INITIALIZED
GUARDRAILS_ACTIVE = LLM_INITIALIZED 

# 1. Create the Pydantic parser
parser = PydanticOutputParser(pydantic_object=GuardrailOutput)

# 2. Create the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are a content moderator for a chatbot. Your job is to detect violations
in a user's message based on three categories:
1.  **Profanity/Slur:** Detects offensive language. If found, set violation_type="profanity" 
    and response="Let's keep our conversation respectful, please."
2.  **Gibberish:** Detects random, nonsensical text (e.g., "asdflkj asdfasdfsadf"). 
    If found, set violation_type="gibberish" and 
    response="I'm sorry, I didn't catch that-could you rephrase?"
3.  **Contradiction:** Detects simple, factual contradictions (e.g., "Is 30C freezing?"). 
    If found, set violation_type="contradiction" and provide a helpful correction 
    (e.g., "No- 30°C is quite warm, not freezing.").

If no violation is found, set is_violation=False, violation_type="none", and response="".

{format_instructions}
"""),
        ("human", "{user_message}")
    ]
)

# 3. Create the LLM Chain (LCEL)
guardrail_chain = prompt_template | llm | parser

def check_llm_guardrails(message: str) -> Optional[str]:
    """
    Checks message using the LLM guardrail chain.
    Returns a specific error response if a violation is found, else None.
    """
    try:
        result: GuardrailOutput = guardrail_chain.invoke({
            "user_message": message,
            "format_instructions": parser.get_format_instructions()
        })
        
        if result.is_violation:
            return result.response
        
        return None
        
    except Exception as e:
        print(f"Error in guardrail check: {e}")
        # Fail safe: if the LLM fails, just let the message pass
        return None