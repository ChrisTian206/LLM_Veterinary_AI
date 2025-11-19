# 🧠 LangGraph Veterinary Assistant

The main interactive AI assistant that provides expert veterinary guidance through conversation. Simply run it in your terminal and start asking questions about your cat's health.

## 🎯 What It Does

- **💬 Chat Interface**: Interactive conversation in your terminal
- **� Image Analysis**: Upload photos of your cat for health assessment
- **� Smart Search**: Finds relevant information from veterinary textbooks
- **🌐 Current Info**: Searches trusted veterinary websites for up-to-date information
- **🚨 Emergency Detection**: Recognizes urgent situations and provides immediate guidance

## 🚀 Quick Start

### Prerequisites
1. **Knowledge Base Ready**: Complete the textbook ingestion first (see `../textbook_to_db/`)
2. **Ollama Models Installed**:
   ```bash
   ollama pull mistral:instruct
   ollama pull qwen3:8b
   ollama pull minicpm-v:8b
   ```
3. **Get Tavily API Key** (for web search):
   - Go to [tavily.com](https://tavily.com) and sign up
   - Get your free API key from the dashboard

### Setup Environment

Create a `.env` file in the project root:
```bash
# In the main LLM_Veterinary_AI directory
touch .env
```

Add your Tavily API key:
```bash
echo "TAVILY_API_KEY=your_api_key_here" >> .env
```

### Run the Assistant

```bash
cd langgraph
python langgraph_flow.py
```

## 💬 How to Use

### Basic Chat
```
Enter your question about your cat: My cat is limping on its front paw
If you want to provide an image: /path/to/cat_photo.jpg (or press Enter to skip)
```

### With Images
```
Enter your question: What's wrong with my cat's ear?
Image path: /Users/yourname/Desktop/cat_ear_photo.jpg
```

### Conversation Flow
- Ask your question
- Optionally provide a photo
- The AI will analyze and respond
- Continue the conversation by answering follow-up questions
- Type `/bye` to exit

## 🔧 Simple Configuration

### If you get memory errors:
```bash
# Close other applications
# Or use smaller models (edit the .py file model names)
```

### If web search isn't working:
```bash
# Check your .env file has the API key
cat .env
# Should show: TAVILY_API_KEY=your_key_here
```

### If models are slow to load:
```bash
# Make sure Ollama is running
ollama serve
```

## � Example Conversation

```bash
$ python langgraph_flow.py

Welcome to the Veterinary Assistant!
You can have a continuous conversation with the AI.
Type '/bye' at any time to exit the conversation.

Enter your question about your cat: My cat has been vomiting and won't eat

If you want to provide an image, enter the file path: 
🤖 I understand you're concerned about your cat vomiting and not eating. 
These can be signs of several conditions. Can you tell me:
- How long has this been going on?
- What does the vomit look like?
- Is your cat drinking water?

Your response: It started yesterday, mostly clear liquid, and yes drinking water

🤖 Based on your description, this could be due to several causes...
[AI provides detailed guidance and follow-up questions]

Your response: /bye

Thank you for using the Veterinary Assistant! Take care of your pet! 🐱
```

## 🐛 Troubleshooting

### "Model not found" error:
```bash
ollama pull mistral:instruct
ollama pull qwen3:8b  
ollama pull minicpm-v:8b
```

### "Knowledge base not found" error:
```bash
# Go back and run the ingestion first
cd ../textbook_to_db
jupyter notebook ingestion.ipynb
```

### Web search not working:
```bash
# Check your .env file in the main directory
ls -la ../.env
# Make sure it contains: TAVILY_API_KEY=your_key_here
```

### Out of memory:
- Close other applications
- Use Activity Monitor (Mac) or Task Manager (Windows) to check RAM usage
- Restart if needed

## ⚡ Performance Tips

- **First run**: Takes longer as models load into memory
- **Subsequent runs**: Much faster once models are cached
- **With images**: Slower processing due to vision model
- **Simple questions**: Usually answered in 10-30 seconds
- **Complex analysis**: May take 1-2 minutes

---

**� Tip**: Start with simple questions to test your setup, then try more complex scenarios with images!
