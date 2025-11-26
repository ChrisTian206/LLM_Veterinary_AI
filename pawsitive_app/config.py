"""
Configuration file for Pawsitive Veterinary AI Assistant
"""

import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent

# Model Configuration
MAIN_MODEL = "ollama:qwen3:8b"
SUMMARIZATION_MODEL = "ollama:llama3.2:3b"
MODEL_TEMPERATURE = 0.5

# Retriever Configuration
# Original collection (for images)
CHROMA_DIRECTORY = str(PROJECT_ROOT.parent / "chroma" / "Cat_Owners_Home_Veterinary_Handbook")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions

# New trimmed collection (for text and tables - better accuracy)
CHROMA_DIRECTORY_TRIMMED = str(PROJECT_ROOT.parent / "chroma" / "TO_Cat_Owners_Home_Veterinary_Handbook_Trimed")
EMBEDDING_MODEL_TRIMMED = "Qwen/Qwen3-Embedding-0.6B"  # 1024 dimensions

ID_KEY = "doc_id"

# Multi-Query Configuration
ENABLE_MULTI_QUERY_TEXTBOOK = True
NUM_TEXTBOOK_QUERIES = 3  # Number of query variations to generate

# Agent Configuration
MAX_CONCURRENT_RESEARCH_UNITS = 3
MAX_RESEARCHER_ITERATIONS = 3
DEFAULT_TEXTBOOK_SEARCH_K = 3
DEFAULT_WEB_SEARCH_MAX_RESULTS = 1

# Memory Configuration
CHECKPOINT_DB_PATH = str(PROJECT_ROOT / "agent_memory.db")

# Tavily Search Configuration
TAVILY_INCLUDE_DOMAINS = [
    "https://www.avma.org/",
    "https://www.merckvetmanual.com/",
    "https://www.vet.cornell.edu/",
    "https://www.vetmed.ucdavis.edu/",
    "https://vcahospitals.com/know-your-pet",
    "https://www.petmd.com/",
    "https://www.aspca.org/pet-care",
    "https://www.petpoisonhelpline.com/",
    "https://www.wormsandgermsblog.com/",
    "https://www.vin.com/"
]

TAVILY_EXCLUDE_DOMAINS = [
    "reddit.com",
    "quora.com",
    "wikipedia.org"
]

# Streamlit UI Configuration
APP_TITLE = "Pawsitive Vet AI"
APP_ICON = "🐾"
PAGE_LAYOUT = "wide"
DEFAULT_THREAD_ID = "default"

# Feature Flags
ENABLE_TOOL_OUTPUT_DISPLAY = False
ENABLE_FILE_DOWNLOAD = False
ENABLE_CONVERSATION_EXPORT = False
