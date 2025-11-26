# 🐾 LLM Veterinary AI

An AI-powered veterinary assistant for cat owners that combines multimodal understanding (text + images) with intelligent reasoning to provide expert guidance for pet health questions, emergency situations, and routine care.

## � What It Does

- **📸 Analyze photos** of your cat alongside text descriptions for health assessment
- **🔍 Search veterinary textbooks** to provide evidence-based guidance  
- **🤖 Engage in conversations** with intelligent follow-up questions
- **🚨 Detect emergencies** and provide appropriate triage guidance
- **🌐 Search trusted websites** for current veterinary information

Built on the authoritative "Cat Owner's Home Veterinary Handbook" by Debra M. Eldredge and supplemented with emergency care, nutrition, and parasite resources.

## 🤖 Models & Embeddings

**Language Models (Ollama):**
- **`qwen3:8b`** - Primary reasoning and conversation model
- **`llama3.2:3b`** - Fast summarization model
- **`minicpm-v:8b`** - Vision model for analyzing cat photos


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

## 🚀 Getting Started

This repository contains multiple implementations of veterinary AI assistants:

### 🐾 **Pawsitive App** (Recommended)
Interactive Streamlit application with dual-collection retrieval and multi-query augmentation.

**→ See [`pawsitive_app/README.md`](pawsitive_app/README.md) for setup and usage**

Features:
- 💬 Multi-turn conversations with persistent memory
- 📚 Dual-collection retrieval (optimized text/tables + images)
- 🔍 Multi-query augmentation for better coverage
- 🌐 Web search integration (Tavily)
- 📝 Context offloading via file management

### 🧠 **Pawsitive V1 Workflow**
Legacy LangGraph-based workflow with multimodal understanding.

**→ See `pawsitive_v1_workflow/` for the original LangGraph implementation**

### 🔬 **Pawsitive V2 ReAct**
Experimental ReAct agent with enhanced reasoning capabilities.

**→ See `pawsitive_v2_ReAct/` for research and development experiments**

## 🔧 Technology Stack

- **LangGraph** - ReAct agent workflow orchestration
- **Streamlit** - Interactive web interface
- **ChromaDB** - Dual-collection vector database (optimized text/tables + images)
- **Ollama** - Local LLM inference
- **HuggingFace Transformers** - Embedding models (BAAI, Qwen)
- **Tavily** - Web search API for current information
- **SQLite** - Persistent conversation memory

## 💡 Key Features

### Dual-Collection Retrieval System

Uses **two optimized vector databases** for better accuracy:

- **Trimmed Collection** (Qwen embeddings, 1024 dims) - Text and tables with higher clinical accuracy
- **Original Collection** (BAAI embeddings, 384 dims) - Image summaries and visual descriptions

### Multi-Query Augmentation

Generates 2-3 query variations per search to improve retrieval coverage and find relevant information from multiple perspectives.

## 🏗️ High-Level Architecture

The system combines local LLM inference (Ollama) with dual-collection vector databases (ChromaDB) for accurate veterinary information retrieval. The ReAct agent orchestrates multiple tools including textbook search, web search, and file management to provide comprehensive responses.

**→ See individual folders for detailed architecture and implementation**

## 📝 Important Notes

- **🏥 Not a replacement** for professional veterinary care
- **🔒 Privacy-focused** - LLM runs locally via Ollama
- **⚠️ Emergency disclaimer** - For serious conditions, consult a veterinarian immediately
- **� Knowledge base** - Based on "Cat Owner's Home Veterinary Handbook" + specialized resources
- **🔄 Continuous improvement** - Multi-query retrieval improves coverage over time

## 📚 Documentation

- **Main App**: See `pawsitive_app/README.md` for detailed setup
- **Retrieval System**: See `unified_retriever.py` for dual-collection implementation
- **Agent Tools**: See `pawsitive_app/tools_and_prompts/` for tool definitions
- **Legacy Workflows**: See `pawsitive_v1_workflow/` and `pawsitive_v2_ReAct/`

---

**📧 Questions?** Open an issue or check the documentation in each folder.
