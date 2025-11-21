"""
Quick test script to verify agent initialization and basic functionality.
"""

from agent_manager import AgentManager

def test_agent():
    """Test agent initialization and simple query."""
    print("🚀 Initializing agent...")
    
    try:
        manager = AgentManager()
        print("✅ Agent initialized successfully!\n")
        
        print("📊 Agent details:")
        print(f"   - Model: {manager.model_name}")
        print(f"   - Tools: {len(manager.all_tools)} total")
        print(f"   - Checkpointer: SQLite at {manager.db_path}")
        print(f"   - Chroma DB: {manager.chroma_directory}\n")
        
        print("💬 Testing simple query...")
        result = manager.invoke(
            message="Hello! What can you help me with?",
            thread_id="test-thread"
        )
        
        # Get last message (agent's response)
        last_message = result["messages"][-1]
        print(f"\n🤖 Agent response:\n{last_message.content}\n")
        
        print("✅ Test passed! Agent is working correctly.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent()
