"""
Agent Manager for Pawsitive Veterinary AI Assistant

Handles agent initialization, configuration, and conversation management.
"""

import os
import sys
import sqlite3
from datetime import datetime
from typing import Optional

import httpx
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from markdownify import markdownify
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing_extensions import Annotated, Literal
import uuid
import base64

# Import local modules
from unified_retriever import UnifiedRetriever
from tools_and_prompts.state import DeepAgentState
from tools_and_prompts.file_tools import ls, read_file, write_file
from tools_and_prompts.todo_tools import write_todos, read_todos
from tools_and_prompts.task_tool import _create_task_tool
from tools_and_prompts.prompts import (
    FILE_USAGE_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS,
    TODO_USAGE_INSTRUCTIONS,
    SUMMARIZE_WEB_SEARCH,
)


class AgentManager:
    """Manages the Pawsitive veterinary AI agent lifecycle and conversations."""
    
    def __init__(
        self,
        model_name: str = "ollama:qwen3:8b",
        summarization_model_name: str = "ollama:llama3.2:3b",
        chroma_directory: str = "../chroma/Cat_Owners_Home_Veterinary_Handbook",
        db_path: str = "agent_memory.db",
    ):
        """Initialize the agent manager with configuration.
        
        Args:
            model_name: Main model for the agent
            summarization_model_name: Model for web search summarization
            chroma_directory: Path to Chroma database
            db_path: Path to SQLite checkpoint database
        """
        # Load environment variables
        load_dotenv()
        
        # Store config
        self.model_name = model_name
        self.summarization_model_name = summarization_model_name
        self.chroma_directory = chroma_directory
        self.db_path = db_path
        
        # Initialize components
        self._init_models()
        self._init_retriever()
        self._init_tavily()
        self._init_tools()
        self._init_checkpointer()
        self._create_agent()
        
    def _init_models(self):
        """Initialize language models."""
        self.model = init_chat_model(model=self.model_name, temperature=0.5)
        self.summarization_model = init_chat_model(
            model=self.summarization_model_name, 
            temperature=0.2
        )
        
    def _init_retriever(self):
        """Initialize textbook retriever with Chroma."""
        text_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        text_vectorstore = Chroma(
            collection_name="text_summaries_and_tables_and_image_summaries",
            embedding_function=text_embeddings,
            persist_directory=self.chroma_directory
        )
        
        text_docstore = Chroma(
            collection_name="text_originals",
            embedding_function=text_embeddings,
            persist_directory=self.chroma_directory
        )
        
        self.retriever = UnifiedRetriever(
            text_vectorstore,
            text_docstore,
            id_key="doc_id"
        )
        
    def _init_tavily(self):
        """Initialize Tavily client for web search."""
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        self.tavily_client = TavilyClient(api_key=api_key)
        self.httpx_client = httpx.Client()
        
    def _init_tools(self):
        """Initialize all agent tools."""
        # Create tool functions with closures to access self
        @tool(parse_docstring=True)
        def textbook_search(
            query: str,
            state: Annotated[DeepAgentState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
            k: Annotated[int, InjectedToolArg] = 3,
        ) -> Command:
            """Search veterinary textbook and save detailed results to files.

            Searches the Cat Owner's Home Veterinary Handbook database for relevant
            information. Saves full textbook content to files for context offloading.
            Returns only essential information to help the agent decide on next steps.

            Args:
                query: Search query to execute on the textbook database
                state: Injected agent state for file storage
                tool_call_id: Injected tool call identifier
                k: Number of results to retrieve (default: 3)

            Returns:
                Command that saves full results to files and provides minimal summary
            """
            results = self.retriever.retrieve_multi_modal(query, k=k)
            
            files = state.get("files", {})
            saved_files = []
            summaries = []
            
            for doc in results:
                modality = doc.get('modality', 'text')
                doc_id = doc.get('doc_id', 'unknown')
                summary = doc.get('summary', '')
                
                # Fetch full original text from docstore
                try:
                    original = self.retriever.text_docstore._collection.get(
                        ids=[doc_id],
                        include=["documents", "metadatas"]
                    )
                    original_text = original["documents"][0] if original["documents"] else summary
                    original_metadata = original["metadatas"][0] if original["metadatas"] else {}
                except Exception:
                    original_text = summary
                    original_metadata = {}
                
                # Generate filename
                uid = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")[:8]
                filename = f"textbook_{modality}_{uid}.md"
                
                # Create file content
                file_content = f"""# Textbook Search Result: {modality.upper()}

**Doc ID:** {doc_id}
**Query:** {query}
**Date:** {self._get_today_str()}
**Type:** {modality}

## Summary
{summary}

## Full Content
{original_text}

## Metadata
{original_metadata}
"""
                
                files[filename] = file_content
                saved_files.append(filename)
                summary_preview = summary[:100] + "..." if len(summary) > 100 else summary
                summaries.append(f"- {filename}: [{modality}] {summary_preview}")
            
            summary_text = f"""📚 Found {len(results)} textbook result(s) for '{query}':

{chr(10).join(summaries)}

Files: {', '.join(saved_files)}
💡 Use read_file() to access full textbook content when needed."""

            return Command(
                update={
                    "files": files,
                    "messages": [
                        {
                            "role": "tool",
                            "content": summary_text,
                            "tool_call_id": tool_call_id,
                        }
                    ],
                }
            )
        
        @tool(parse_docstring=True)
        def tavily_search(
            query: str,
            state: Annotated[DeepAgentState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
            max_results: Annotated[int, InjectedToolArg] = 1,
            topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
        ) -> Command:
            """Search web and save detailed results to files while returning minimal context.

            Performs web search and saves full content to files for context offloading.
            Returns only essential information to help the agent decide on next steps.

            Args:
                query: Search query to execute
                state: Injected agent state for file storage
                tool_call_id: Injected tool call identifier
                max_results: Maximum number of results to return (default: 1)
                topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')

            Returns:
                Command that saves full results to files and provides minimal summary
            """
            search_results = self._run_tavily_search(query, max_results, topic)
            processed_results = self._process_search_results(search_results, query)
            
            files = state.get("files", {})
            saved_files = []
            summaries = []
            
            for result in processed_results:
                filename = result['filename']
                
                file_content = f"""# Search Result: {result['title']}

**URL:** {result['url']}
**Query:** {query}
**Date:** {self._get_today_str()}

## Summary
{result['summary']}

## Raw Content
{result['raw_content'] if result['raw_content'] else 'No raw content available'}
"""
                
                files[filename] = file_content
                saved_files.append(filename)
                summaries.append(f"- {filename}: {result['summary']}...")
            
            summary_text = f"""🔍 Found {len(processed_results)} result(s) for '{query}':

{chr(10).join(summaries)}

Files: {', '.join(saved_files)}
💡 Use read_file() to access full details when needed."""

            return Command(
                update={
                    "files": files,
                    "messages": [
                        {
                            "role": "tool",
                            "content": summary_text,
                            "tool_call_id": tool_call_id,
                        }
                    ],
                }
            )
        
        @tool(parse_docstring=True)
        def think_tool(reflection: str) -> str:
            """Tool for strategic reflection on research progress and decision-making.

            Use this tool after each search to analyze results and plan next steps systematically.
            This creates a deliberate pause in the research workflow for quality decision-making.

            When to use:
            - After receiving search results: What key information did I find?
            - Before deciding next steps: Do I have enough to answer comprehensively?
            - When assessing research gaps: What specific information am I still missing?
            - Before concluding research: Can I provide a complete answer now?
            - How complex is the question: Have I reached the number of search limits?

            Reflection should address:
            1. Analysis of current findings - What concrete information have I gathered?
            2. Gap assessment - What crucial information is still missing?
            3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
            4. Strategic decision - Should I continue searching or provide my answer?

            Args:
                reflection: Your detailed reflection on research progress, findings, gaps, and next steps

            Returns:
                Confirmation that reflection was recorded for decision-making
            """
            return f"Reflection recorded: {reflection}"
        
        # Store tools
        self.textbook_search = textbook_search
        self.tavily_search = tavily_search
        self.think_tool = think_tool
        
        # Organize tools
        self.sub_agent_tools = [textbook_search, tavily_search, think_tool]
        self.built_in_tools = [ls, read_file, write_file, write_todos, read_todos, think_tool]
        
        # Create research sub-agent config
        research_sub_agent = {
            "name": "research-agent",
            "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
            "prompt": RESEARCHER_INSTRUCTIONS.format(date=self._get_today_str()),
            "tools": ["textbook_search", "tavily_search", "think_tool"],
        }
        
        # Create task delegation tool
        task_tool = _create_task_tool(
            self.sub_agent_tools,
            [research_sub_agent],
            self.model,
            DeepAgentState
        )
        
        self.delegation_tools = [task_tool]
        self.all_tools = self.sub_agent_tools + self.built_in_tools + self.delegation_tools
        
    def _init_checkpointer(self):
        """Initialize SQLite checkpointer for persistent memory."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(conn)
        
    def _create_agent(self):
        """Create the ReAct agent with all tools and configuration."""
        # Build instructions
        max_concurrent_research_units = 3
        max_researcher_iterations = 3
        
        subagent_instructions = SUBAGENT_USAGE_INSTRUCTIONS.format(
            max_concurrent_research_units=max_concurrent_research_units,
            max_researcher_iterations=max_researcher_iterations,
            date=self._get_today_str(),
        )
        
        instructions = (
            "# TODO MANAGEMENT\n"
            + TODO_USAGE_INSTRUCTIONS
            + "\n\n"
            + "=" * 80
            + "\n\n"
            + "# FILE SYSTEM USAGE\n"
            + FILE_USAGE_INSTRUCTIONS
            + "\n\n"
            + "=" * 80
            + "\n\n"
            + "# SUB-AGENT DELEGATION\n"
            + subagent_instructions
        )
        
        # Create agent
        self.agent = create_react_agent(
            self.model,
            self.all_tools,
            prompt=instructions,
            state_schema=DeepAgentState,
            checkpointer=self.checkpointer
        )
        
    def _get_today_str(self) -> str:
        """Get current date in human-readable format."""
        return datetime.now().strftime("%a %b %-d, %Y")
    
    def _run_tavily_search(
        self,
        query: str,
        max_results: int = 1,
        topic: Literal["general", "news", "finance"] = "general",
    ) -> dict:
        """Execute Tavily search with veterinary domain filtering."""
        return self.tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_domains=[
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
            ],
            exclude_domains=["reddit.com", "quora.com", "wikipedia.org"],
            include_raw_content=True
        )
    
    def _summarize_webpage(self, content: str) -> BaseModel:
        """Summarize webpage content using AI."""
        class Summary(BaseModel):
            filename: str = Field(description="Name of the file to store.")
            summary: str = Field(description="Key learnings from the webpage.")
        
        try:
            structured_model = self.summarization_model.with_structured_output(Summary)
            return structured_model.invoke([
                HumanMessage(content=SUMMARIZE_WEB_SEARCH.format(
                    webpage_content=content,
                    date=self._get_today_str()
                ))
            ])
        except Exception:
            return Summary(
                filename="search_result.md",
                summary=content[:1000] + "..." if len(content) > 1000 else content
            )
    
    def _process_search_results(self, results: dict, query: str) -> list[dict]:
        """Process Tavily search results with summarization."""
        processed_results = []
        
        for result in results.get('results', []):
            url = result['url']
            
            try:
                response = self.httpx_client.get(url)
                if response.status_code == 200:
                    raw_content = markdownify(response.text)
                    summary_obj = self._summarize_webpage(raw_content)
                else:
                    raw_content = result.get('raw_content', '')
                    summary_obj = self._summarize_webpage(raw_content)
            except Exception:
                raw_content = result.get('raw_content', '')
                summary_obj = self._summarize_webpage(raw_content)
            
            # Uniquify filenames
            uid = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")[:8]
            name, ext = os.path.splitext(summary_obj.filename)
            summary_obj.filename = f"{name}_{uid}{ext}"
            
            processed_results.append({
                'url': result['url'],
                'title': result['title'],
                'summary': summary_obj.summary,
                'filename': summary_obj.filename,
                'raw_content': raw_content,
            })
        
        return processed_results
    
    def invoke(self, message: str, thread_id: str = "default") -> dict:
        """Invoke the agent with a message.
        
        Args:
            message: User message
            thread_id: Conversation thread identifier
            
        Returns:
            Agent response with messages and state
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config
        )
        
        return result
    
    def stream(self, message: str, thread_id: str = "default"):
        """Stream agent responses.
        
        Args:
            message: User message
            thread_id: Conversation thread identifier
            
        Yields:
            Agent response chunks with incremental updates
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        for chunk in self.agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            stream_mode="updates"
        ):
            yield chunk
    
    def get_state(self, thread_id: str = "default") -> dict:
        """Get current conversation state.
        
        Args:
            thread_id: Conversation thread identifier
            
        Returns:
            Current state snapshot
        """
        config = {"configurable": {"thread_id": thread_id}}
        return self.agent.get_state(config)
