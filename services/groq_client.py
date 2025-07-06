import os
from langchain_groq import ChatGroq
from utils.logger import get_logger

logger = get_logger(__name__)

def get_llm():
    try:
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama3-70b-8192")
        )
        logger.info("Groq LLM initialized")
        return llm
    except Exception as e:
        logger.error(f"Groq init failed: {e}")
        raise
