# Document Selection Enhancement Notes

## Overview
Notes for implementing LLM-as-judge document selection to improve prompt length management and document relevance in the veterinary AI system.

## Current Issue
- Fixed limit of 5 documents from `relevant_docs[:5]` 
- No intelligent selection based on medical relevance
- Prompt length can exceed 10k tokens
- All documents treated equally regardless of diagnostic value

## Proposed Solution: LLM-as-Judge Document Selection

### Implementation Plan

#### 1. Add New State Field
```python
class GraphState(TypedDict):
    # ... existing fields ...
    selected_docs: Optional[List[Dict[str, Any]]]  # New field for LLM-selected docs
```

#### 2. Create Smart Document Selector Node
```python
def smart_document_selector(state):
    """Use LLM-as-judge to intelligently select top 5 documents"""
    docs = state.get("relevant_docs", [])
    query = state.get("refined_query", "")
    search_context = state.get("search_context", "")
    latest_user_input = state.get("latest_user_input", "")
    
    if len(docs) <= 5:
        return {"selected_docs": docs}
    
    # Create compact document descriptions for ranking
    doc_summaries = []
    for i, doc in enumerate(docs):
        modality = doc.get('modality') or (doc.get('original_metadata') or {}).get('type')
        summary = doc.get('summary', '')[:300]  # Truncate for efficiency
        
        if modality in ('image', 'image_summary'):
            doc_summaries.append(f"{i+1}. [IMAGE] {summary}")
        elif modality == 'table':
            doc_summaries.append(f"{i+1}. [TABLE] {summary}")
        else:
            doc_summaries.append(f"{i+1}. [TEXT] {summary}")
    
    ranking_prompt = f"""You are a veterinary document selector. Rank these {len(docs)} documents by relevance.

Query: {query}
Specific need: {search_context}
Latest user input: {latest_user_input}

Documents:
{chr(10).join(doc_summaries)}

Select the 5 most relevant document numbers for comprehensive veterinary guidance.
Focus on: symptoms, diagnostics, treatment, safety, visual aids.

Respond with exactly 5 numbers separated by commas (e.g., "2,7,1,9,4")."""

    # Use fast model for ranking
    response = ollama.chat(
        model="mistral:instruct",
        messages=[{"role": "user", "content": ranking_prompt}],
        options={"temperature": 0}
    )
    
    # Parse response and select documents
    # ... implementation details ...
    
    return {"selected_docs": selected_docs}
```

#### 3. Update Graph Flow
```python
def build_graph():
    # Add new document selector node
    builder.add_node("document_selector", smart_document_selector)
    
    # Modified flow:
    # relevancy_check → document_selector → thinking
    builder.add_edge("relevancy_check", "document_selector")
    builder.add_edge("document_selector", "thinking")
```

#### 4. Update Thinking Node
```python
def thinking_node(state):
    # Use selected docs instead of relevant_docs[:5]
    selected_docs = state.get("selected_docs", [])
    
    prompt += f"\nHere are {len(selected_docs)} carefully selected documents from veterinary handbook:\n"
    # ... rest of implementation
```

### Key Benefits

✅ **Intelligent Selection**: LLM judges medical relevance, diagnostic value, and visual importance
✅ **Prompt Length Control**: Always exactly 5 best documents, preventing token overflow  
✅ **Medical Focus**: Prioritizes symptoms, diagnostics, treatments, and safety information
✅ **Visual Context**: Maintains important image/diagram selection for veterinary diagnosis
✅ **Context Awareness**: Considers user query, search context, and conversation history

### LLM Judge Criteria
The ranking LLM will prioritize documents based on:
1. **Symptom specificity** - Documents matching exact symptoms described
2. **Diagnostic procedures** - Step-by-step diagnostic guidance
3. **Treatment protocols** - Actionable management steps
4. **Safety considerations** - Emergency signs and when to seek help
5. **Visual aids** - Images/diagrams that support understanding

### Alternative: Multi-LLM Approach
If single prompt still too long, could split into:
- **LLM 1**: Medical Analyzer (extracts key insights from documents)
- **LLM 2**: Response Synthesizer (creates user-friendly responses)

### Implementation Priority
- **Phase 1**: Implement basic LLM-as-judge document selection
- **Phase 2**: Add adaptive selection based on query complexity
- **Phase 3**: Consider multi-LLM architecture if needed

## Files to Modify
- `langgraph_flow.py`: Add document selector node and update graph flow
- Test with various veterinary scenarios to ensure quality selection

## Notes
- Use `mistral:instruct` for fast document ranking
- Keep fallback to first 5 docs if LLM ranking fails
- Monitor prompt lengths to ensure staying under 10k tokens
- Consider caching document rankings for repeated queries