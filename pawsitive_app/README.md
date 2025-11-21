# 🐾 Pawsitive Veterinary AI Assistant

A professional Streamlit-based chat interface for an AI veterinary assistant powered by LangGraph, Ollama, and RAG (Retrieval-Augmented Generation).

## Features

- 💬 **Multi-Conversation Management**: Handle multiple concurrent conversations with separate threads
- 🔄 **Real-Time Streaming**: Watch responses generate in real-time
- 📚 **Textbook Knowledge**: Search veterinary textbooks (Cat Owner's Home Veterinary Handbook)
- 🌐 **Web Research**: Real-time web search with domain filtering for veterinary sources
- 📝 **File Management**: Agent can create, read, and manage files
- ✅ **Task Management**: TODO tracking and task delegation
- 💾 **Persistent Memory**: All conversations saved to SQLite database
- 🎨 **Professional UI**: Clean, responsive Streamlit interface

## Quick Start

### 1. Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- [Ollama](https://ollama.ai/) with models:
  - `qwen3:8b` (main agent)
  - `llama3.2:3b` (summarization)
- Tavily API key (for web search)

### 2. Installation

All dependencies are already installed via `uv`. If you need to reinstall:

```bash
cd /Users/mas/Desktop/LLM_Veterinary_AI/pawsitive_app
uv sync
```

### 3. Environment Setup

Make sure `.env` file exists with:

```env
TAVILY_API_KEY=your_tavily_api_key_here
```

### 4. Run the App

```bash
uv run streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Architecture

### Components

1. **agent_manager.py**: Core agent logic
   - Initializes LangGraph ReAct agent
   - Manages tools (textbook search, web search, file operations)
   - Handles conversation state and checkpointing

2. **app.py**: Streamlit UI
   - Chat interface
   - Conversation management
   - Real-time streaming display
   - State visualization

3. **config.py**: Configuration
   - Model settings
   - Paths and directories
   - Feature flags

4. **unified_retriever.py**: RAG retrieval
   - Multi-modal retrieval from Chroma
   - Text and image support

5. **tools_and_prompts/**: Agent tools and prompts
   - File operations (ls, read_file, write_file)
   - TODO management (write_todos, read_todos)
   - Task delegation
   - Strategic thinking tool
   - System prompts

### Data Flow

```
User Input → Streamlit UI → AgentManager → LangGraph Agent
                                              ↓
                                         [Tools]
                                    ↙        ↓        ↘
                          Textbook Search  Web Search  File Ops
                                    ↓        ↓        ↓
                                    Chroma   Tavily   Memory
                                              ↓
                                         Response
                                              ↓
                                    Streamlit UI (streaming)
```

## Usage Examples

### Basic Question
```
User: "What are the symptoms of feline leukemia?"
Agent: [Searches textbook] → Provides detailed answer with sources
```

### Research Task
```
User: "Compare dry food vs wet food for cats"
Agent: [Searches textbook + web] → Comprehensive comparison with latest research
```

### Complex Workflow
```
User: "Create a nutrition guide for senior cats"
Agent: 
1. Creates TODO list
2. Delegates research tasks
3. Searches textbook and web
4. Compiles findings into files
5. Synthesizes final guide
```

## Configuration

Edit `config.py` to customize:

- **Models**: Change LLM models
- **Retriever**: Update Chroma paths or embedding models
- **Search**: Modify domain filters for web search
- **Features**: Enable/disable tool outputs, file downloads

## Project Structure

```
pawsitive_app/
├── app.py                      # Streamlit UI
├── agent_manager.py            # Agent logic
├── config.py                   # Configuration
├── test_agent.py              # Test script
├── unified_retriever.py       # RAG retrieval
├── tools_and_prompts/         # Agent tools
│   ├── __init__.py
│   ├── file_tools.py
│   ├── prompts.py
│   ├── research_tools.py
│   ├── state.py
│   ├── task_tool.py
│   └── todo_tools.py
├── data/                      # Runtime data (created automatically)
├── agent_memory.db           # SQLite checkpoint database
├── .env                      # Environment variables
├── pyproject.toml           # uv project config
└── README.md                # This file
```

## Troubleshooting

### Agent initialization fails
- Ensure Ollama is running: `ollama list`
- Check models are installed: `ollama pull qwen3:8b` and `ollama pull llama3.2:3b`

### Chroma retrieval errors
- Verify Chroma database exists at: `../chroma/Cat_Owners_Home_Veterinary_Handbook`
- Check collections exist: should have `text_summaries_and_tables_and_image_summaries` and `text_originals`

### Web search not working
- Verify `TAVILY_API_KEY` in `.env`
- Test API key: `curl -X POST https://api.tavily.com/search -H "Content-Type: application/json" -d '{"api_key":"YOUR_KEY","query":"test"}'`

### Streamlit errors
- Clear cache: `uv run streamlit cache clear`
- Reset memory: Delete `agent_memory.db` and restart

## Development

### Testing Agent
```bash
uv run python test_agent.py
```

### Running Streamlit in Dev Mode
```bash
uv run streamlit run app.py --server.runOnSave true
```

### Adding New Tools
1. Define tool in `agent_manager.py` or `tools_and_prompts/`
2. Add to tool lists in `agent_manager.py`
3. Update prompts in `tools_and_prompts/prompts.py`

## License

MIT

## Disclaimer

⚠️ This AI assistant provides general information only. Always consult a licensed veterinarian for specific medical advice.
