# 📚 Textbook to Database Ingestion Pipeline

This module processes veterinary PDF textbooks into a searchable multimodal knowledge base using ChromaDB. It extracts text, images, and tables, creates AI-generated summaries, and builds vector embeddings for semantic search.

## 🎯 What It Does

- **📄 PDF Parsing**: Extracts text, images, and tables using Unstructured AI
- **🤖 AI Summarization**: Creates semantic summaries using Ollama models
- **🔍 Vector Embeddings**: Generates embeddings for text and images
- **💾 Database Storage**: Stores everything in ChromaDB for fast retrieval
- **🖼️ Image Processing**: Handles medical diagrams and illustrations

## 🏗️ Architecture Overview

```
PDF Input → Unstructured Parsing → AI Summarization → Vector Embeddings → ChromaDB Storage
    ↓              ↓                    ↓                 ↓               ↓
  Raw PDF      Text/Images/Tables   Semantic Summaries  Search Vectors  Queryable DB
```

## 📋 System Requirements

- **RAM**: 16GB minimum (for vision models and LLM processing)
- **Storage**: ~5GB for models + 2-5GB per processed textbook
- **Ollama Models**: Ensure required models are installed

## 🚀 Quick Start

### 1. Set Up Configuration

Edit the configuration variables in `ingestion.ipynb`:

```python
# Input Configuration
pdf_file = './data/Cat_Owners_Home_Veterinary_Handbook_Trimed.pdf'
image_output_dir = './figures/Cat_Owners_Home_Veterinary_Handbook'
chroma_persist_dir = './chroma/Cat_Owners_Home_Veterinary_Handbook/'
```

### 2. Run the Ingestion Pipeline

```bash
cd textbook_to_db
jupyter notebook ingestion.ipynb
```

Execute all cells in sequence. The process takes **15-30 minutes** depending on PDF size and hardware.

## ⚙️ Configuration Options

### Input/Output Paths

```python
# Source PDF to process
pdf_file = './data/your_textbook.pdf'

# Where to save extracted images
image_output_dir = './figures/your_textbook_name'

# Where to save the vector database
chroma_persist_dir = './chroma/your_textbook_name/'
```

### Processing Parameters

```python
# Text filtering and chunking
window_size = 2                    # Context window for text chunks
min_meaningful_text_length = 75    # Minimum text length to process

# Image processing
extract_images_in_pdf = True       # Extract embedded images
infer_table_structure = True       # Parse tables
```

### Memory Management

```python
# Batch processing for large PDFs
batch_size = 10                    # Process elements in batches
max_image_size = (1024, 1024)     # Resize large images
```

## 📁 File Structure

```
textbook_to_db/
├── ingestion.ipynb              # Main ingestion notebook
├── textbook_loading.py          # Core processing functions
├── unified_retriever.py         # Multimodal retrieval system
└── README.md                    # This file
```

## 🔧 Core Components

### `textbook_loading.py`

**Key Functions:**
- `load_book()` - PDF parsing and element extraction
- `clean_and_categorize_elements()` - Filters and organizes content
- `summarize_elements()` - AI-powered summarization
- `store_in_chromadb()` - Vector database storage

### `unified_retriever.py`

**Retrieval System:**
- Combines text and image search
- Supports multimodal queries
- Handles metadata filtering

### `ingestion.ipynb`

**Processing Pipeline:**
1. PDF extraction and parsing
2. Content categorization
3. AI summarization
4. Vector embedding creation
5. Database storage

## 📖 Adding New Textbooks

### Step 1: Prepare Your PDF
```bash
# Place your PDF in the data directory
cp your_textbook.pdf ../data/
```

### Step 2: Configure Paths
```python
pdf_file = './data/your_textbook.pdf'
image_output_dir = './figures/your_textbook'
chroma_persist_dir = './chroma/your_textbook/'
```

### Step 3: Run Ingestion
Execute all cells in `ingestion.ipynb`. Monitor memory usage during processing.

### Step 4: Verify Results
```python
# Check extracted elements
print(f"Found {len(texts)} text chunks")
print(f"Found {len(tables)} tables") 
print(f"Found {len(images_raw)} images")
```

## 🛠️ Models and Dependencies

### Required Ollama Models
```bash
# For text summarization and processing
ollama pull mistral:instruct

# For image analysis (vision model)
ollama pull minicpm-v:8b
```

### Key Dependencies
- **Unstructured**: PDF parsing and element extraction
- **ChromaDB**: Vector database for storage
- **OpenCLIP**: Image embeddings
- **HuggingFace**: Text embeddings
- **Ollama**: Local LLM inference

## 📊 Performance Considerations

### Memory Usage
- **Text Processing**: ~2-4GB RAM
- **Image Processing**: ~4-8GB RAM (with vision model)
- **Vector Storage**: ~1-2GB RAM per 1000 pages

### Processing Time
- **Small PDF (50 pages)**: ~5-10 minutes
- **Medium PDF (200 pages)**: ~15-25 minutes  
- **Large PDF (500+ pages)**: ~30-60 minutes

### Storage Requirements
- **Images**: ~100-500MB per textbook
- **ChromaDB**: ~200-1GB per textbook
- **Models**: ~4-6GB (one-time download)

## 🐛 Troubleshooting

### Memory Issues
```bash
# If you encounter OOM errors:
# 1. Reduce batch size
# 2. Process smaller PDF sections
# 3. Close other applications
# 4. Consider using GPU if available
```

### Model Loading Issues
```bash
# Ensure Ollama is running
ollama serve

# Verify models are installed
ollama list
```

### Permission Errors
```bash
# Ensure write permissions for output directories
chmod -R 755 figures/ chroma/
```

## 🔍 Output Verification

After successful ingestion, you should see:

```
chroma/your_textbook/
├── chroma.sqlite3                 # Main database
├── text_summaries_and_tables_and_image_summaries/  # Text vectors
├── text_originals/                # Original text chunks
├── images/                        # Image vectors  
└── image_originals/               # Original images

figures/your_textbook/
├── figure-1-1.jpg                 # Extracted images
├── figure-2-3.png
└── ...
```

## 🚀 Next Steps

After successful ingestion:
1. **Test Retrieval**: Use `unified_retriever.py` to test searches
2. **Run LangGraph**: Navigate to `../langgraph/` and run the main application
3. **Add More Books**: Repeat process for additional textbooks

## 📝 Notes

- **Reprocessing**: Delete the ChromaDB directory to completely reprocess a textbook
- **Incremental Updates**: Currently not supported - full reprocessing required
- **GPU Acceleration**: Optional but recommended for faster processing
- **Internet Required**: For downloading models and some dependencies

---

**💡 Tip**: Start with smaller PDF sections to test your setup before processing large textbooks.
