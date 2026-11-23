from __future__ import annotations

import logging
import os

from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize the Gemini model, adjust temperature as needed, 0.0 is deterministic and 2.0 is more creative
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, google_api_key=GEMINI_API_KEY)


@tool
def search_google(query: str) -> str:
    """Search Google for latest information on topic."""
    logger.info(f"Searching Google for: {query}")
    return gemini_model.invoke(query, tools=[GenAITool(google_search={})]).content


graph = create_react_agent(
    gemini_model,
    tools=[search_google],
    prompt="You are a helpful assistant. Always answer in a funny way",
)
