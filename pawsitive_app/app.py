"""
Pawsitive Veterinary AI Assistant - Streamlit App

A professional chat interface for the veterinary AI agent with:
- Multi-conversation management
- Real-time streaming responses
- File and todo tracking
- Persistent memory
"""

import streamlit as st
from agent_manager import AgentManager
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Pawsitive Vet AI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .tool-message {
        background-color: #fff3e0;
        font-size: 0.9rem;
    }
    .stButton button {
        width: 100%;
    }
    @keyframes blink {
        0%, 20% { content: '.'; }
        40% { content: '..'; }
        60%, 100% { content: '...'; }
    }
    .thinking-animation::after {
        content: '.';
        animation: blink 1.5s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "agent_manager" not in st.session_state:
    with st.spinner("🚀 Initializing Pawsitive AI Agent..."):
        st.session_state.agent_manager = AgentManager()
    st.success("✅ Agent initialized!")

if "current_thread" not in st.session_state:
    st.session_state.current_thread = "default"

if "show_tool_outputs" not in st.session_state:
    st.session_state.show_tool_outputs = False

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "selected_file" not in st.session_state:
    st.session_state.selected_file = None

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

if "processing_placeholders" not in st.session_state:
    st.session_state.processing_placeholders = None

# Sidebar
with st.sidebar:
    st.markdown("## 🐾 Pawsitive Vet AI")
    st.markdown("*Your AI Veterinary Assistant*")
    st.markdown("---")
    
    # Conversation management
    st.markdown("### 💬 Conversations")
    
    # Get list of threads from agent state (source of truth)
    try:
        # Get all threads from checkpointer
        cursor = st.session_state.agent_manager.checkpointer.conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id;")
        db_threads = [row[0] for row in cursor.fetchall()]
        
        # Always include default thread and current thread
        conversation_names = ["default"]
        if st.session_state.current_thread not in conversation_names:
            conversation_names.append(st.session_state.current_thread)
        
        # Add other threads from database
        for thread in db_threads:
            if thread not in conversation_names:
                conversation_names.append(thread)
        
    except Exception:
        conversation_names = ["default"]
        if st.session_state.current_thread not in conversation_names:
            conversation_names.append(st.session_state.current_thread)
    
    # Thread selector
    current_index = conversation_names.index(st.session_state.current_thread) if st.session_state.current_thread in conversation_names else 0
    
    selected_thread = st.selectbox(
        "Select conversation:",
        conversation_names,
        index=current_index,
        disabled=st.session_state.is_processing
    )
    
    if selected_thread != st.session_state.current_thread:
        st.session_state.current_thread = selected_thread
        st.rerun()
    
    # New conversation button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New Chat", disabled=st.session_state.is_processing):
            new_thread = f"thread-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            st.session_state.current_thread = new_thread
            st.rerun()
    
    with col2:
        if st.button("🗑️ Delete", disabled=st.session_state.is_processing or st.session_state.current_thread == "default"):
            # Delete from database
            try:
                cursor = st.session_state.agent_manager.checkpointer.conn.cursor()
                cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?;", (st.session_state.current_thread,))
                st.session_state.agent_manager.checkpointer.conn.commit()
            except Exception as e:
                st.error(f"Failed to delete: {e}")
            
            st.session_state.current_thread = "default"
            st.rerun()
    
    st.markdown("---")
    
    # Display conversation state
    st.markdown("### 📊 Current State")
    
    try:
        state = st.session_state.agent_manager.get_state(st.session_state.current_thread)
        
        # Files - make them clickable
        files = state.values.get("files", {})
        if files:
            with st.expander(f"📁 Files ({len(files)})", expanded=False):
                for filename in files.keys():
                    if st.button(f"📄 {filename}", key=f"file_{filename}"):
                        st.session_state.selected_file = filename
        else:
            st.text("📁 Files: None yet")
        
        # Todos
        todos = state.values.get("todos", [])
        if todos:
            with st.expander(f"✅ TODOs ({len(todos)})", expanded=False):
                for todo in todos:
                    status_icons = {
                        "pending": "⏳",
                        "in_progress": "🔄",
                        "completed": "✅"
                    }
                    status_icon = status_icons.get(todo.get("status"), "❓")
                    content = todo.get('content', 'No description')
                    st.text(f"{status_icon} {content}")
        else:
            st.text("✅ TODOs: None")
        
        # Message count
        messages = state.values.get("messages", [])
        st.text(f"💬 Messages: {len(messages)}")
        
    except Exception as e:
        st.text("📊 State: Not available")
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    st.session_state.show_tool_outputs = st.checkbox(
        "Show tool outputs",
        value=st.session_state.show_tool_outputs,
        disabled=st.session_state.is_processing,
        help="Disable during processing to avoid interruptions"
    )
    
    # Info
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Pawsitive Vet AI** combines:
    - 📚 Veterinary textbook knowledge
    - 🌐 Real-time web research
    - 🧠 Strategic reasoning
    - 📝 File & task management
    
    Powered by LangGraph + Ollama
    """)

# Main content area
st.markdown('<div class="main-header">🐾 Pawsitive Veterinary AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Expert veterinary guidance powered by AI</div>', unsafe_allow_html=True)

# Show processing indicator
if st.session_state.is_processing:
    st.info("⏳ Agent is processing... Please wait before making changes to settings or switching conversations.")

# File viewer modal using dialog
@st.dialog("📄 File Viewer", width="large")
def show_file_modal(filename, content):
    """Display file content in a modal dialog."""
    st.markdown(f"**Filename:** `{filename}`")
    st.markdown("---")
    
    # Display content in a scrollable container
    st.markdown(content)
    
    st.markdown("---")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="⬇️ Download File",
            data=content,
            file_name=filename,
            mime="text/markdown",
            key=f"download_{filename}",
            use_container_width=True
        )
    with col2:
        if st.button("✖️ Close", key="close_modal", use_container_width=True):
            st.session_state.selected_file = None
            st.rerun()

# Trigger modal if file is selected
if st.session_state.selected_file:
    try:
        state = st.session_state.agent_manager.get_state(st.session_state.current_thread)
        files = state.values.get("files", {})
        
        if st.session_state.selected_file in files:
            file_content = files[st.session_state.selected_file]
            show_file_modal(st.session_state.selected_file, file_content)
        else:
            st.error(f"File not found: {st.session_state.selected_file}")
            st.session_state.selected_file = None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.session_state.selected_file = None

# Helper function to load messages from agent state
def load_messages_from_state(thread_id: str):
    """Load messages from agent's actual state (source of truth)."""
    try:
        state = st.session_state.agent_manager.get_state(thread_id)
        messages = state.values.get("messages", [])
        
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, "type"):
                if msg.type == "human":
                    formatted_messages.append({
                        "role": "user",
                        "content": msg.content
                    })
                elif msg.type == "ai":
                    formatted_messages.append({
                        "role": "assistant",
                        "content": msg.content
                    })
                elif msg.type == "tool" and st.session_state.show_tool_outputs:
                    formatted_messages.append({
                        "role": "tool",
                        "content": msg.content
                    })
        
        return formatted_messages
    except Exception as e:
        return []

# Display chat history
chat_container = st.container()

with chat_container:
    # Load messages from agent state (source of truth)
    current_messages = load_messages_from_state(st.session_state.current_thread)
    
    if not current_messages:
        st.info("👋 Hello! I'm your veterinary AI assistant. Ask me anything about pet health, nutrition, or care!")
    else:
        for msg in current_messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(content)
            
            elif role == "assistant":
                with st.chat_message("assistant", avatar="🐾"):
                    st.markdown(content)
            
            elif role == "tool":
                with st.chat_message("assistant", avatar="🔧"):
                    with st.expander("Tool Output", expanded=False):
                        st.markdown(content)
    
    # Create a placeholder for processing status - this stays in chat container
    processing_status = st.empty()

# Chat input and stop button
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.chat_input(
        "Ask me anything about veterinary care...",
        disabled=st.session_state.is_processing,
        key="chat_input"
    )

with col2:
    if st.session_state.is_processing:
        if st.button("⏹️ Stop", key="stop_button", use_container_width=True, type="primary"):
            st.session_state.stop_requested = True
            st.warning("⚠️ Stop requested - agent will stop at next checkpoint...")

if user_input:
    # Set processing flag and reset stop flag
    st.session_state.is_processing = True
    st.session_state.stop_requested = False
    
    # Display user message immediately in the chat container
    with chat_container:
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
    
    full_response = ""
    tools_used = []
    files_created = []
    
    try:
        # Show initial thinking status with animated dots (below the user message)
        with processing_status.container():
            st.markdown('<div style="padding: 1rem; background-color: #e3f2fd; border-radius: 0.5rem; margin-bottom: 1rem;"><span style="font-size: 1.1rem;">🤔 <span class="thinking-animation">Thinking</span></span></div>', unsafe_allow_html=True)
        
        # Stream agent response with updates mode
        for chunk in st.session_state.agent_manager.stream(
            user_input,
            thread_id=st.session_state.current_thread
        ):
            # Check if stop was requested
            if st.session_state.stop_requested:
                with processing_status.container():
                    st.warning("⏹️ Stopped by user")
                break
            
            # chunk is a dict with node name as key
            for node_name, node_output in chunk.items():
                if "messages" in node_output:
                    messages = node_output["messages"]
                    if messages:
                        last_msg = messages[-1]
                        
                        # Show real-time progress based on message types
                        if hasattr(last_msg, "type"):
                            if last_msg.type == "ai" and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                # Agent is calling tools - show what it's doing
                                tool_names = [tc["name"] for tc in last_msg.tool_calls]
                                
                                # Track tools used
                                for tool_name in tool_names:
                                    if tool_name not in tools_used:
                                        tools_used.append(tool_name)
                                
                                if "task" in tool_names:
                                    current_status = "📋 Planning tasks and delegating to sub-agents..."
                                elif "textbook_search" in tool_names:
                                    current_status = "📚 Searching veterinary textbook..."
                                elif "tavily_search" in tool_names:
                                    current_status = "🌐 Searching the web for latest information..."
                                elif "write_todos" in tool_names:
                                    current_status = "✅ Creating TODO list..."
                                elif "write_file" in tool_names:
                                    current_status = "📝 Writing to file..."
                                elif "read_file" in tool_names:
                                    current_status = "📖 Reading file..."
                                elif "think_tool" in tool_names:
                                    current_status = "💭 Reflecting on findings..."
                                else:
                                    current_status = f"🔧 Using tool: {', '.join(tool_names)}"
                                
                                # Update status with animated dots
                                with processing_status.container():
                                    st.markdown(f'<div style="padding: 1rem; background-color: #e3f2fd; border-radius: 0.5rem; margin-bottom: 1rem;"><span style="font-size: 1.1rem;"><span class="thinking-animation">{current_status}</span></span></div>', unsafe_allow_html=True)
                            
                            elif last_msg.type == "tool":
                                # Tool just returned results
                                tool_name = getattr(last_msg, 'name', 'unknown')
                                
                                # Check if files were mentioned in tool output
                                if hasattr(last_msg, 'content') and 'Files:' in last_msg.content:
                                    # Extract file names from tool output
                                    import re
                                    file_matches = re.findall(r'([a-zA-Z0-9_\-]+\.md)', last_msg.content)
                                    files_created.extend(file_matches)
                                
                                if tool_name == "textbook_search":
                                    current_status = "✅ Textbook search complete - analyzing results..."
                                elif tool_name == "tavily_search":
                                    current_status = "✅ Web search complete - processing information..."
                                elif tool_name == "task":
                                    current_status = "✅ Sub-agent task complete - synthesizing..."
                                else:
                                    current_status = f"✅ {tool_name} complete - processing..."
                                
                                # Update status with animated dots
                                with processing_status.container():
                                    st.markdown(f'<div style="padding: 1rem; background-color: #e3f2fd; border-radius: 0.5rem; margin-bottom: 1rem;"><span style="font-size: 1.1rem;"><span class="thinking-animation">{current_status}</span></span></div>', unsafe_allow_html=True)
                            
                            elif last_msg.type == "ai" and hasattr(last_msg, "content") and last_msg.content:
                                # AI response with content - capture it
                                full_response = last_msg.content
        
        # Clear processing status
        processing_status.empty()
        
        # Display the agent's response immediately (before rerun)
        if full_response:
            with chat_container:
                with st.chat_message("assistant", avatar="🐾"):
                    st.markdown(full_response)
                    
                    # Show tools used summary
                    if tools_used or files_created:
                        with st.expander("🔍 Research Details", expanded=False):
                            if tools_used:
                                st.write("**🔧 Tools Used:**")
                                tool_icons = {
                                    "textbook_search": "📚",
                                    "tavily_search": "🌐",
                                    "think_tool": "💭",
                                    "task": "📋",
                                    "write_file": "📝",
                                    "read_file": "📖",
                                    "write_todos": "✅",
                                }
                                for tool in tools_used:
                                    icon = tool_icons.get(tool, "🔧")
                                    st.text(f"  {icon} {tool}")
                            
                            if files_created:
                                st.write(f"\n**📄 Files Created:** {len(set(files_created))}")
                                for f in set(files_created):
                                    st.text(f"  • {f}")
        else:
            with chat_container:
                with st.chat_message("assistant", avatar="🐾"):
                    st.warning("No response generated. Please try again.")
        
    except Exception as e:
        processing_status.empty()
        # Show error in processing status area
        with processing_status.container():
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("🔍 Debug Info - Click to expand"):
                st.code(traceback.format_exc())
                st.write("Thread ID:", st.session_state.current_thread)
                st.write("User Input:", user_input)
    
    finally:
        # Always reset processing flag and stop flag
        st.session_state.is_processing = False
        st.session_state.stop_requested = False
    
    # Rerun to update UI and load fresh messages from agent state
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    <p>⚠️ <strong>Disclaimer:</strong> This AI assistant provides general information only. 
    Always consult a licensed veterinarian for specific medical advice.</p>
</div>
""", unsafe_allow_html=True)
