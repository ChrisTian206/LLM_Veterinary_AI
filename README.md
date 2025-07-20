# 🐾 Veterinary AI Assistant for Pet Owners

A comprehensive AI-powered veterinary assistant that helps pet owners with at-home diagnosis, step-by-step procedures, and veterinary guidance. This system combines multimodal understanding (text + images) with intelligent reasoning capabilities to provide accurate, contextual veterinary advice.

## 🎯 Purpose

This AI assistant is designed to help pet owners:
- **Ask veterinary questions** and receive expert-level guidance
- **Perform at-home diagnosis** with visual and textual analysis
- **Get step-by-step procedures** for various pet care scenarios
- **Think through complex situations** using reasoning and tool-calling capabilities
- **Retrieve relevant information** from veterinary textbooks and resources
- **Ask clarifying questions** when more context is needed

## 🏗️ System Architecture

### Core Components

#### 1. **Multimodal Knowledge Base**
- **Text Processing**: Extracts and chunks text from veterinary PDFs using semantic chunking
- **Image Analysis**: Processes images with vision-language models for detailed descriptions
- **Table Extraction**: Captures structured data from veterinary tables
- **Context Enrichment**: Links images and tables with surrounding text for better understanding

#### 2. **Intelligent Retrieval System**
- **Unified Retriever**: Combines text and image search in a shared vector space
- **OpenCLIP Embeddings**: Creates multimodal embeddings for text and images
- **ChromaDB Storage**: Persistent vector database with separate collections for summaries and originals
- **Semantic Search**: Finds relevant content across all modalities

#### 3. **Reasoning & Tool-Calling Engine**
- **LangGraph Workflow**: Orchestrates complex reasoning processes
- **Multi-Path Routing**: 
  - **Path A**: Deep investigation for health problems requiring detailed analysis
  - **Path B**: Direct Q&A with optional clarification loops
- **Tool Integration**: Can call external tools for additional information or user interaction
- **Conversational Intelligence**: Asks clarifying questions when needed

#### 4. **Vision-Language Processing**
- **Image Summarization**: Converts visual information to searchable text
- **Query Refinement**: Enhances user queries with visual context
- **Multimodal Reranking**: Improves result relevance using cross-modal understanding

## 🚀 Key Features

### 🤖 Intelligent Reasoning
- **Chain-of-Thought Processing**: Breaks down complex problems step-by-step
- **Tool Calling**: Can retrieve additional information or ask user questions
- **Context Awareness**: Understands when more information is needed
- **Multi-Step Analysis**: Handles complex veterinary scenarios

### 📸 Multimodal Understanding
- **Image Analysis**: Processes pet photos for health assessment
- **Text + Image Queries**: Combines visual and textual information
- **Visual Diagnosis**: Helps identify symptoms from photos
- **Contextual Retrieval**: Finds relevant images and text together

### 💬 Conversational Intelligence
- **Clarification Loops**: Asks for more details when queries are ambiguous
- **Progressive Refinement**: Builds understanding through conversation
- **Personalized Responses**: Adapts to specific pet situations
- **Emergency Awareness**: Recognizes urgent situations

### 📚 Comprehensive Knowledge Base
- **Veterinary Textbooks**: Multiple authoritative sources
- **Emergency Procedures**: Step-by-step guidance for urgent situations
- **Nutrition Information**: Dietary advice and recommendations
- **Parasite Management**: Prevention and treatment protocols
- **Ear Care**: Specialized knowledge for ear-related issues

## 📁 Project Structure

```
LLM_Veterinary_AI/
├── 📚 data/                          # Veterinary PDF sources
│   ├── Cat_Owners_Home_Veterinary_Handbook_*.pdf
│   ├── EmergencyInfectiveDisease.pdf
│   ├── Nutrition_20Pgs.pdf
│   └── PARASITES.pdf
├── 🔧 core/
│   ├── textbook_loading.py          # PDF processing and chunking
│   ├── unified_retriever.py         # Multimodal retrieval system
│   └── ingestion.ipynb              # Knowledge base construction
├── 🧠 langgraph designs/            # Reasoning workflow designs
│   ├── tri_routing_description.txt  # Multi-path routing logic
│   └── graph_structure1.txt         # Workflow architecture
├── 🔬 experiment/                   # Testing and development
│   ├── query+img.ipynb              # Multimodal query processing
│   ├── langgraph exp/               # LangGraph experiments
│   └── *.jpg                        # Test images
├── 🖼️ figures/                      # Extracted images from PDFs
├── 💾 chroma/                       # Vector database storage
├── 🤖 llava/                        # Vision model integration
└── 📋 requirements.txt              # Dependencies
```

## 🛠️ Technology Stack

### Core Dependencies
- **LangChain & LangGraph**: Workflow orchestration and reasoning
- **ChromaDB**: Vector database for multimodal storage
- **OpenCLIP**: Multimodal embeddings for text and images
- **Ollama**: Local LLM inference (supports various models)
- **Transformers**: NLP and vision models
- **Unstructured**: PDF parsing and element extraction

### AI Models
- **Vision Models**: LLaVA, MiniCPM-V, Llama3.2-Vision
- **Language Models**: Llama3.2, DeepSeek, Qwen variants
- **Embedding Models**: OpenCLIP, Sentence Transformers
- **Reranking**: Cross-encoder models for improved relevance

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Ollama installed with appropriate models
- Sufficient RAM for multimodal processing (8GB+ recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd LLM_Veterinary_AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Ollama models**
   ```bash
   # Install vision model
   ollama pull llama3.2-vision:11b-instruct-q4_K_M
   
   # Install language model
   ollama pull llama3.2:3b
   ```

4. **Build the knowledge base**
   ```bash
   # Run the ingestion notebook to process PDFs
   jupyter notebook ingestion.ipynb
   ```

### Usage Examples

#### Basic Text Query
```python
query = "What are the signs of ear infection in cats?"
# System processes query through reasoning pipeline
# Returns relevant text and images with step-by-step guidance
```

#### Multimodal Query with Image
```python
query = "What's wrong with my cat?"
image_path = "cat_ear_problem.jpg"
# System analyzes image, refines query, retrieves relevant information
# Provides diagnosis and treatment recommendations
```

#### Emergency Scenario
```python
query = "My cat is having trouble breathing"
# System recognizes emergency, provides immediate steps
# Offers escalation guidance and when to seek veterinary care
```

## 🔄 Workflow Examples

### 1. **Simple Q&A Path**
```
User: "How long are cats pregnant?"
→ Direct retrieval → Answer generation → Response
```

### 2. **Clarification Path**
```
User: "What's the best food for my cat?"
→ Retrieval → Clarity check → Ask: "How old is your cat?"
→ User: "She's 3 years old"
→ Enhanced retrieval → Specific answer
```

### 3. **Complex Investigation Path**
```
User: "My cat seems sick" + [photo]
→ Image analysis → Query refinement → Multi-step retrieval
→ Tool calls for additional info → Progressive diagnosis
→ Step-by-step guidance → Follow-up recommendations
```

## 🎯 Use Cases

### 🏠 At-Home Care
- **Symptom Assessment**: Analyze photos and descriptions
- **First Aid Guidance**: Emergency procedures and immediate steps
- **Preventive Care**: Nutrition, grooming, and health maintenance
- **Behavioral Issues**: Understanding and addressing pet behavior

### 🚨 Emergency Situations
- **Triage Assessment**: Determine urgency of veterinary care
- **Immediate Actions**: What to do while seeking professional help
- **Emergency Procedures**: Step-by-step guidance for critical situations
- **Veterinary Preparation**: Information to share with veterinarians

### 📋 Routine Care
- **Vaccination Schedules**: Timing and importance of vaccinations
- **Parasite Prevention**: Flea, tick, and worm prevention
- **Nutrition Guidance**: Age-appropriate diets and feeding schedules
- **Grooming Procedures**: Proper care for different coat types

## 🔧 Development

### Adding New Knowledge Sources
1. Place PDF files in `data/` directory
2. Run `ingestion.ipynb` to process new sources
3. The system automatically integrates new content

### Customizing Models
- Modify model configurations in `requirements.txt`
- Update Ollama model references in notebooks
- Adjust embedding models in `unified_retriever.py`

### Extending Workflows
- Design new LangGraph workflows in `langgraph designs/`
- Implement new reasoning paths for specific scenarios
- Add specialized tools for veterinary procedures

## 📝 Notes

- **Re-ingest** knowledge base when adding new PDFs or changing processing logic
- **Model Selection**: Choose models based on your hardware capabilities
- **Emergency Disclaimer**: This system provides guidance but doesn't replace professional veterinary care
- **Data Privacy**: All processing happens locally; no data is sent to external services

## 🤝 Contributing

This project welcomes contributions for:
- Additional veterinary knowledge sources
- Improved reasoning workflows
- Enhanced multimodal processing
- Better user interaction patterns

## 📄 License

MIT License - See LICENSE file for details

---

**⚠️ Important**: This AI assistant provides educational information and guidance but should not replace professional veterinary care. Always consult with a licensed veterinarian for serious health concerns or emergencies.
