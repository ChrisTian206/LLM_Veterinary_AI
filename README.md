# 🐾 LLM Veterinary AI

An AI-powered veterinary assistant for cat owners that combines multimodal understanding (text + images) with intelligent reasoning to provide expert guidance for pet health questions, emergency situations, and routine care.

## � What It Does

- **📸 Analyze photos** of your cat alongside text descriptions for health assessment
- **🔍 Search veterinary textbooks** to provide evidence-based guidance  
- **🤖 Engage in conversations** with intelligent follow-up questions
- **🚨 Detect emergencies** and provide appropriate triage guidance
- **🌐 Search trusted websites** for current veterinary information

Built on the authoritative "Cat Owner's Home Veterinary Handbook" by Debra M. Eldredge and supplemented with emergency care, nutrition, and parasite resources.

## 🤖 Ollama Models Used

- **`mistral:instruct`** - Primary reasoning and conversation model
- **`qwen3:8b`** - Advanced decision-making and complex reasoning  
- **`minicpm-v:8b`** - Vision model for analyzing cat photos
- **`BAAI/bge-small-en-v1.5`** - Text embeddings for document search
- **`ViT-g-14` (OpenCLIP)** - Image embeddings for visual search

## 📁 Folder Structure

```
LLM_Veterinary_AI/
├── 📚 data/                                    # Source veterinary PDFs
│   ├── Cat_Owners_Home_Veterinary_Handbook_*.pdf
│   ├── EmergencyInfectiveDisease.pdf
│   ├── Ears.pdf, Nutrition.pdf, PARASITES.pdf
│   └── ...
├── 🧠 langgraph/                              # Main reasoning engine
│   ├── langgraph_flow.py                     # Complete LangGraph workflow
│   └── graph_visualization.png               # Workflow diagram
├── � textbook_to_db/                         # Knowledge base creation
│   ├── ingestion.ipynb                       # Main ingestion notebook
│   ├── textbook_loading.py                   # PDF processing utilities
│   └── unified_retriever.py                  # Multimodal retrieval system
├── �️ chroma/                                  # Vector database storage
│   ├── Cat_Owners_Home_Veterinary_Handbook/   # Main knowledge base
│   ├── Ears/, EmergencyInfectiveDisease/      # Specialized collections
│   └── ...
├── �️ figures/                                 # Extracted images from PDFs
├── 📷 Kaggle_PetDiseaseImages/                # Disease image dataset
├── 👁️ llava/                                  # Vision model examples
└── 📋 requirements.txt, nb.ipynb, README.md
```

## 🚀 Quick Start


### Setup

1. **Install dependencies**
   ```bash
   git clone https://github.com/ChrisTian206/LLM_Veterinary_AI.git
   cd LLM_Veterinary_AI
   pip install -r requirements.txt
   ```

2. **Install Ollama models**
   ```bash
   ollama pull mistral:instruct
   ollama pull qwen3:8b
   ollama pull minicpm-v:8b
   ```

### Usage

1. **First: Load the textbooks** (required before running the graph)
   ```bash
   cd textbook_to_db
   jupyter notebook ingestion.ipynb
   # Execute all cells to build the knowledge base (~15-30 minutes)
   ```

2. **Then: Run the veterinary assistant**
   ```bash
   cd ../langgraph
   python langgraph_flow.py
   ```

3. **Chat with the AI**
   - Enter your cat health questions
   - Optionally provide image paths for visual analysis
   - Type `/bye` to exit

## 🔧 Technology Stack

- **LangGraph** - Workflow orchestration and reasoning
- **ChromaDB** - Vector database for multimodal search
- **Ollama** - Local LLM inference
- **OpenCLIP** - Multimodal embeddings
- **Unstructured** - PDF parsing and extraction

## 💡 Example Usage

```bash
# User input
"My cat has been scratching its ear a lot and shaking its head"

# Optional: Provide image
"path/to/cat_ear_photo.jpg"

# AI Process:
1. Analyzes image (if provided) → describes visible symptoms
2. Refines query → "cat ear scratching head shaking possible infection"
3. Searches knowledge base → finds relevant ear condition info
4. Asks clarifying questions → "Is there any discharge or odor?"
5. Provides guidance → step-by-step care instructions
```

## 📝 Important Notes

- **🏥 Not a replacement** for professional veterinary care
- **🔒 Privacy-focused** - All processing happens locally
- **⚠️ Emergency disclaimer** - For serious conditions, consult a veterinarian immediately
- **🔄 Re-run ingestion** when adding new PDFs to the knowledge base

---

**📧 Questions?** Check the detailed documentation in `langgraph/` and `textbook_to_db/` folders.
