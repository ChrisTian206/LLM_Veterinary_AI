from typing import Any, Optional
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
import re
import os
import uuid
#Jojo stepped on my keyboard
#w23eqerdw 
import json

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document

from unified_retriever import UnifiedRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

class Element(BaseModel):
    type: str
    text: Any
    context: Optional[str] = None
    original_index: Optional[int] = None

def load_book(file_name):
    """
    Loads a PDF book and partitions its elements (text, tables) using Unstructured.
    
    Args:
        file_name (str): The path to the PDF file.
                                         
    Returns:
        list: A list of raw elements extracted from the PDF.
    """
    # Get elements
    raw_pdf_elements = partition_pdf(
        filename=file_name,
        languages=['eng'],
        strategy='hi_res',
        extract_images_in_pdf=False,
        infer_table_structure=True,
    )
    return raw_pdf_elements

def is_junk_text(text):
    t = text.strip()
    if not t or len(t) < 5:
        return True
    junk_patterns = [
        r'^—o—$', r'^\*+$', r'^_+$', r'^page \d+', r'^\d{1,2}/\d{1,2}/\d{2,4}$',
        r"^cat owner[’']?s home veterinary handbook$", r'^\d+$', r'^ch\d+', r'^fig(ure)?[-\d]+',
        r'^04_095300.*page.*$', r'^[\W_]+$'
    ]
    for pat in junk_patterns:
        if re.match(pat, t, re.IGNORECASE):
            return True
    # If mostly non-alphabetic
    if len(t) > 0 and sum(c.isalpha() for c in t) < 0.3 * len(t):
        return True
    return False

def clean_and_categorize_elements(raw_pdf_elements, min_meaningful_text_length=15):
    """
    Cleans and categorizes raw PDF elements into texts and tables.

    Args:
        raw_pdf_elements (list): A list of raw elements obtained from partition_pdf.
        min_meaningful_text_length (int, optional): Minimum length for a text block to be considered meaningful. Defaults to 15.

    Returns:
        tuple: (texts, tables)
            - texts (list): List of cleaned text chunks.
            - tables (list): List of extracted table contents.
    """
    text_for_semantic_chunking = []
    tables_raw = []

    current_text_block = ""
    current_context_prefix = ""

    def finalize_text_block_inner(last_index=None):
        nonlocal current_text_block, current_context_prefix
        cleaned = current_text_block.strip()
        if cleaned and len(cleaned) >= min_meaningful_text_length and not is_junk_text(cleaned):
            text_for_semantic_chunking.append(Element(type="text", text=cleaned, original_index=last_index))
        current_text_block = ""

    for i, element in enumerate(raw_pdf_elements):
        element_type_str = str(type(element))
        element_text = str(element).strip()

        if "unstructured.documents.elements.Header" in element_type_str or \
           "unstructured.documents.elements.Title" in element_type_str:
            finalize_text_block_inner(i)
            if not is_junk_text(element_text):
                current_context_prefix = element_text + " "
        elif "unstructured.documents.elements.NarrativeText" in element_type_str or \
             "unstructured.documents.elements.ListItem" in element_type_str or \
             "unstructured.documents.elements.Text" in element_type_str:
            if len(element_text) < 5 and not any(char.isalpha() for char in element_text):
                continue
            if not current_text_block and current_context_prefix:
                current_text_block += current_context_prefix
            current_text_block += element_text + " "
        elif "unstructured.documents.elements.Table" in element_type_str:
            finalize_text_block_inner(i)
            if not is_junk_text(element_text):
                tables_raw.append(Element(type="table", text=element_text, original_index=i))
        # Ignore Image, FigureCaption, Footer, etc.
        else:
            continue

    finalize_text_block_inner(i)
    texts = text_for_semantic_chunking
    tables = tables_raw
    return texts, tables

def enrich_table_context(tables, all_raw_elements, window_size=1):
    """
    Enriches the context for each table by looking at a window of surrounding text and captions.

    Args:
        tables (list): A list of table Element objects to enrich.
        all_raw_elements (list): All raw elements from the PDF, used to find surrounding text.
        window_size (int, optional): Number of elements to look before and after the table. Defaults to 1.
    """
    for tbl_element in tables:
        tbl_index = tbl_element.original_index
        if tbl_index is None:
            continue
        start_index = max(0, tbl_index - window_size)
        end_index = min(len(all_raw_elements), tbl_index + window_size + 1)
        surrounding_text_elements = []
        for j in range(start_index, end_index):
            surrounding_element = all_raw_elements[j]
            element_type_str = str(type(surrounding_element))
            element_text = str(surrounding_element).strip()
            if "unstructured.documents.elements.NarrativeText" in element_type_str or \
               "unstructured.documents.elements.ListItem" in element_type_str or \
               "unstructured.documents.elements.Text" in element_type_str or \
               "unstructured.documents.elements.FigureCaption" in element_type_str:
                if len(element_text) >= 5 or any(char.isalpha() for char in element_text):
                    surrounding_text_elements.append(element_text)
            if "unstructured.documents.elements.Header" in element_type_str or \
               "unstructured.documents.elements.Title" in element_type_str:
                lower_element_text = element_text.lower()
                is_running_header = (
                    "qxp" in lower_element_text or "pm" in lower_element_text or
                    "am" in lower_element_text or "page" in lower_element_text or
                    re.search(r'\\d{1,2}/\\d{1,2}/\\d{2,4}', lower_element_text)
                )
                if not is_running_header:
                    surrounding_text_elements.append(element_text)
        enriched_context = " ".join(surrounding_text_elements).strip()
        if not enriched_context:
            enriched_context = "No specific text context available around this table."
        tbl_element.context = enriched_context

def semantic_chunk_texts(texts, embedding_model=None):
    """
    Combines all text chunks into one and applies semantic chunking using LangChain.
    Returns a list of semantically meaningful text chunks.
    """
    combined_text = " ".join([t.text if hasattr(t, "text") else str(t) for t in texts])

    embedding_model = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
    text_splitter = SemanticChunker(embedding_model)
    docs = text_splitter.create_documents([combined_text])
    semantic_chunks = [doc.page_content for doc in docs]
    return semantic_chunks

def summarize_texts(texts):
    """
    Semantically chunk the input texts, then summarize each chunk using Ollama models.
    Returns a list of summaries.
    """
    # Semantic chunking
    print("Performing semantic chunking on texts...")
    semantic_chunks = semantic_chunk_texts(texts)
    print(f"Created {len(semantic_chunks)} semantic chunks from {len(texts)} text elements")
    
    model = ChatOllama(model="llama3.2:3b")
    prompt_text_summary = (
        "You are an assistant tasked with concisely summarizing text sections related to veterinary advice and pet care. "
        "Focus on key information, main ideas, and any actionable advice. Just give me the summary, be concise and do not be verbose. Text chunk: {element} "
    )
    prompt_text = ChatPromptTemplate.from_template(prompt_text_summary)
    text_summarize_chain = {"element": lambda x: x} | prompt_text | model | StrOutputParser()
    
    print(f"Summarizing {len(semantic_chunks)} chunks with Ollama (concurrency=8)...")
    return text_summarize_chain.batch(semantic_chunks, {"max_concurrency": 8})

def summarize_tables(tables, raw_pdf_elements=None):
    """
    Summarizes table elements using Ollama models, enriching context if raw_pdf_elements is provided.
    Returns a list of table summaries.
    """
    if not tables:
        print("No tables to summarize")
        return []
    
    if raw_pdf_elements is not None and tables and hasattr(tables[0], 'context'):
        enrich_table_context(tables, raw_pdf_elements, window_size=1)
    
    model = ChatOllama(model="llama3.2:3b")
    prompt_table_summary = (
        "You are an assistant tasked with extracting key information, trends, and important numerical data from the provided table, "
        "especially as it relates to veterinary topics, animal health, or clinical practice. Use the provided context to help interpret the table. "
        "Just give me the summary, be concise and do not be verbose.\n\n"
        "Context: {context}\n"
        "Table chunk: {element}"
    )
    prompt_table = ChatPromptTemplate.from_template(prompt_table_summary)
    table_summarize_chain = (
        {"element": lambda x: x["element"], "context": lambda x: x["context"]}
        | prompt_table
        | model
        | StrOutputParser()
    )
    table_context_pairs = []
    for tbl in tables:
        context = getattr(tbl, 'context', "")
        table_context_pairs.append({"element": tbl.text if hasattr(tbl, 'text') else tbl, "context": context})
    
    print(f"Summarizing {len(table_context_pairs)} tables with Ollama (concurrency=8)...")
    return table_summarize_chain.batch(table_context_pairs, {"max_concurrency": 8})

def summarize_elements(texts, tables, raw_pdf_elements=None):
    """
    Summarizes text and table elements using Ollama models.
    Returns:
        tuple: (text_summaries, table_summaries)
    """
    # Summarize texts
    text_summaries = summarize_texts(texts)
    # Summarize tables
    table_summaries = summarize_tables(tables, raw_pdf_elements)
    print("Texts and Tables Summary Done!")
    return text_summaries, table_summaries

def store_in_chromadb(text_summaries, texts, table_summaries, tables, persist_directory="./chroma_db", batch_size=5):
    """
    Stores the summarized text and table data into ChromaDB vector stores on disk.
    - Text vectorstore/docstore: for text and table content, using a text embedding model.
    - Uses batch processing to avoid GPU memory issues
    Returns a UnifiedRetriever
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    import uuid, json
    import gc
    import torch
    from unified_retriever import UnifiedRetriever

    # Clear GPU memory before starting
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    print("🧹 Cleared GPU memory before embedding")

    # Use MPS (GPU) for embedding with aggressive memory management
    text_embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'device': 'mps'},  # Use Apple Silicon GPU
        encode_kwargs={'batch_size': 4}  # Small internal batch size to reduce memory
    )

    # Vectorstores and docstores
    text_vectorstore = Chroma(
        collection_name="text_summaries_and_tables",
        embedding_function=text_embeddings,
        persist_directory=persist_directory
    )
    text_docstore = Chroma(
        collection_name="text_originals",
        embedding_function=text_embeddings,
        persist_directory=persist_directory
    )
    id_key = "doc_id"

    # Store text chunks (summaries and originals) in batches
    if texts:
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=s, metadata={id_key: doc_ids[i], "type": "text"})
            for i, s in enumerate(text_summaries)
        ]
        original_text_docs = [
            Document(page_content=texts[i].text, metadata={id_key: doc_ids[i], "type": "text"})
            for i in range(len(texts))
        ]
        
        # Batch processing for summaries
        print(f"Adding {len(summary_texts)} text summaries in batches of {batch_size}...")
        for i in range(0, len(summary_texts), batch_size):
            batch_docs = summary_texts[i:i+batch_size]
            batch_ids = doc_ids[i:i+batch_size]
            text_vectorstore.add_documents(batch_docs, ids=batch_ids)
            print(f"  Processed batch {i//batch_size + 1}/{(len(summary_texts)-1)//batch_size + 1}")
            
            # Clear GPU memory after EVERY batch to prevent OOM
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
        
        # Batch processing for originals
        print(f"Adding {len(original_text_docs)} original texts in batches of {batch_size}...")
        for i in range(0, len(original_text_docs), batch_size):
            batch_docs = original_text_docs[i:i+batch_size]
            batch_ids = doc_ids[i:i+batch_size]
            text_docstore.add_documents(batch_docs, ids=batch_ids)
            print(f"  Processed batch {i//batch_size + 1}/{(len(original_text_docs)-1)//batch_size + 1}")
            
            # Clear GPU memory after EVERY batch to prevent OOM
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
    
    # Store tables as JSON (summaries and originals) in batches
    if tables:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_tables = [
            Document(page_content=s, metadata={id_key: table_ids[i], "type": "table"})
            for i, s in enumerate(table_summaries)
        ]
        original_table_docs = [
            Document(page_content=json.dumps(tables[i].dict()), metadata={id_key: table_ids[i], "type": "table"})
            for i in range(len(tables))
        ]
        
        # Batch processing for table summaries
        print(f"Adding {len(summary_tables)} table summaries in batches of {batch_size}...")
        for i in range(0, len(summary_tables), batch_size):
            batch_docs = summary_tables[i:i+batch_size]
            batch_ids = table_ids[i:i+batch_size]
            text_vectorstore.add_documents(batch_docs, ids=batch_ids)
            print(f"  Processed batch {i//batch_size + 1}/{(len(summary_tables)-1)//batch_size + 1}")
            
            # Clear GPU memory after EVERY batch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
        
        # Batch processing for original tables
        print(f"Adding {len(original_table_docs)} original tables in batches of {batch_size}...")
        for i in range(0, len(original_table_docs), batch_size):
            batch_docs = original_table_docs[i:i+batch_size]
            batch_ids = table_ids[i:i+batch_size]
            text_docstore.add_documents(batch_docs, ids=batch_ids)
            print(f"  Processed batch {i//batch_size + 1}/{(len(original_table_docs)-1)//batch_size + 1}")
            
            # Clear GPU memory after EVERY batch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
    
    # Final aggressive cleanup
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    print("✅ All documents stored successfully! GPU memory cleared.")

    return UnifiedRetriever(text_vectorstore, text_docstore, None, None, id_key)



