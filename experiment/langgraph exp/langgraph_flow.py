import sys
import os
import uuid
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))
import ollama
import re
import json

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict
from typing import Optional, List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_chroma import Chroma
from unified_retriever import UnifiedRetriever

class GraphState(TypedDict):
    text_query: str
    image_path: Optional[str]
    query_type : str
    refined_query: Optional[str]
    queries_for_retrieval: Optional[List[str]]
    current_sub_query: Optional[str]
    retrieved_docs: Optional[List[Dict[str, Any]]]
    reranked_docs: Optional[List[Dict[str, Any]]]
    relevant_docs: Optional[List[Dict[str, Any]]]
    followup_questions: Optional[List[str]]
    user_responses: Optional[List[str]]
    loop_count: int
    hypotheses: Optional[List[str]]
    next_action: Optional[str]
    user_actions: Optional[List[str]]
    intermediate_thoughts: Optional[List[str]]
    generated_answer: Optional[str]
    hallucination_check: Optional[bool]
    answer_sufficient: Optional[bool]
    emergency_instructions: Optional[str]
    emergency_retrieved_docs: Optional[List[Dict[str, Any]]]
    web_search_results: Optional[List[Dict[str, Any]]]
    final_answer: Optional[str]
    path_taken: Optional[List[str]]
    error: Optional[str]
    
def query_handler(state):
    text_query = state.get("text_query", "")
    image_path = state.get("image_path", None)
    prompt = (
        "You are a domain classifier for a veterinary assistant. "
        "If an image is provided, understand the image from veterinary point of view."
        "A user query is the combination of text query and image(if there is). "
        "Then, classify the user query into one of three categories:\n"
        "1. 'emergency' — If the user query is about a veterinary emergency (e.g., mass bleeding, serious bone fracture, unconsciousness, severe breathing difficulty, or other life-threatening situations).\n"
        "2. 'Q&A' — If the user query is about is about general veterinary questions, symptom checks, or non-emergency animal health issues.\n\n"
        "3. 'irrelevant' — If the user query is NOT about veterinary, animal health, pet care, etc.\n"
        "Your response must be exactly one of: 'irrelevant', 'emergency', or 'Q&A'. Do not explain your answer or add anything else.\n\n"
        f"User input: {text_query}\n"
    )
    messages = [{
        "role": "user",
        "content": prompt,
        "images": []
    }]
    if image_path and os.path.exists(image_path):
        messages[0]["images"].append(image_path)
    response = ollama.chat(
        model="minicpm-v:8b",
        messages=messages,
        options={"temperature": 0.2}
    )
    result = response['message']['content'].strip().lower()
    if result not in ['irrelevant', 'emergency', 'q&a']:
        result = 'irrelevant'

    print(f"1️⃣ Query type determined: {result}")
    return {"query_type": result}

def get_image_summary(image_path):
    prompt = """From a feline veterinary stand point, provide a highly detailed and objective \
                description of the image. Focus on all observable elements, actions, \
                objects, subjects, their attributes (e.g., color, size, texture), \
                their spatial relationships, and any discernible context or implied scene. \
                Also focus on all possible health issue.                
                Describe any text present in the image. This description must be exhaustive \
                and purely factual, capturing every significant visual detail to serve as a \
                comprehensive textual representation for further analysis by another AI model. \
                If the image is entirely irrelevant or contains no discernible subject, \
                state "No relevant visual information.""" .strip()
    messages = [{
        "role": "user",
        "content": prompt,
        "images": [image_path]
    }]
    response = ollama.chat(
        model="minicpm-v:8b",
        messages=messages,
        options={"temperature": 0.2}
    )
    return response['message']['content']

def query_refinement_node(state):
    text_query = state.get("text_query", "")
    image_path = state.get("image_path", None)
    image_summary = get_image_summary(image_path) if image_path else ""
    if image_summary:
        prompt = (
        "You are a veterinary assistant AI. Your task is to rewrite and expand the user's question about their cat to make it more effective for searching a veterinary knowledge base.\n\n"
        "You are NOT being asked to give medical advice, make a diagnosis, or recommend treatments.\n\n"
        "Use the image description only to clarify the concern, but **do not invent or assume** any details (such as environment, causes, or severity) not explicitly mentioned by the user or image.\n\n"
        "The refined query must:\n"
        "- Accurately represent the user's concern about their cat\n"
        "- Factually describe any symptoms or visible signs\n"
        "- Include open-ended questions about **possible causes**, **diagnostic steps**, and **general management or prevention**\n"
        "- Remain **neutral and open-ended**, avoiding assumptions or conclusions, stay grounded to user query.\n"
        "- Be phrased as **a single paragraph**, clear and concise, suitable for search retrieval\n"
        "- Do not add new symptoms, behaviors, or environmental details unless they appear in the user query or image description.\n"
        "- Output **only** the refined query — no introductions or explanations\n\n"
        "Here are examples:\n"
        "---\n"
        "User query: My cat keeps shaking its head a lot.\n"
        "Image description: Redness and dark wax visible in one ear.\n"
        "Refined query: My cat has been shaking its head frequently, and I've noticed redness and dark wax in one ear. I'd like to understand what might be causing these symptoms, what diagnostic steps are typically used to evaluate ear conditions in cats, and what general management or preventive options may apply.\n"
        "---\n"
        "User query: My cat has been throwing up for two days.\n"
        "Image description: Pile of partially digested food on carpet.\n"
        "Refined query: My cat has been vomiting for the past two days, with piles of partially digested food. I want to explore potential causes of vomiting in cats, how to tell if it's serious, what diagnostic approaches are used, and general advice for managing this before seeing a vet.\n"
        "---\n"
        f"User query: {text_query}\n"
        f"Image description: {image_summary}\n"
        "Refined query:"
    )
    else:
        prompt = (
        "You are a veterinary assistant AI. Your task is to rewrite and expand the user's question about their cat to make it more effective for searching a veterinary knowledge base.\n\n"
        "You are NOT being asked to give medical advice, make a diagnosis, or recommend treatments.\n\n"
        "The refined query must:\n"
        "- Clearly describe the user's concern about their cat\n"
        "- Remain **neutral and open-ended**, avoiding assumptions or conclusions, stay grounded to user query.\n"
        "- Include helpful questions about **possible causes**, **diagnostic considerations**, and **general management or prevention**\n"
        "- Be phrased as **a single paragraph**, clear and concise, with no extra fluff\n"
        "Do not add new symptoms, behaviors, or environmental details unless they appear in the user query or image description."
        "- Output **only** the refined query — no introductions or explanations\n\n"
        "Here are examples:\n"
        "---\n"
        "User query: I think my cat has a fever, its nose is hot.\n"
        "Refined query: I'm concerned my cat may have a fever because its nose feels hotter than usual. I want to understand what can cause fever in cats, what signs to look for, and what general steps I should take before consulting a veterinarian.\n"
        "---\n"
        "User query: My cat's been sleeping more than usual and not eating.\n"
        "Refined query: My cat is sleeping much more than usual and has lost interest in eating. I'd like to know what potential causes could lead to these symptoms, how to assess if it's urgent, and what general steps I can take before visiting a vet.\n"
        "---\n"
        f"User query: {text_query}\n"
        "Refined query:"
    )
    messages = [{
        "role": "user",
        "content": prompt
    }]
    response = ollama.chat(
        model="mistral:instruct", 
        messages=messages,
        options={"temperature": 0}
    )

    print("2️⃣ Query refined: ", response['message']['content'][0:100], "...")
    return {"refined_query": response['message']['content']}

def query_decomposition(state):
    refined_query = state['refined_query']
    query_decomposition_prompt = ChatPromptTemplate.from_template("""You are a veterinary knowledge assistant. Your task is to break down the following refined query into a list of **concise, semantically rich phrases**, each representing a **different aspect** of the query.

        Guidelines:
        - Each phrase should represent a **distinct, relevant concept** or question implied in the user's concern.
        - Focus on **causes, symptoms, diagnostics, management steps, risk factors, and context**.
        - Avoid repeating or overlapping ideas.
        - Do NOT include general fluff or irrelevant topics.
        - Use phrases or noun-like expressions (not full sentences).
        - Aim for maximum relevance to the original refined query — each phrase should help retrieve part of a comprehensive answer.
        - At the end, include 2-3 **visually grounded search phrases** that require images, diagrams, or visual aids to understand.

        Output only a **JSON array of strings**, with no extra text or explanation.

        Refined query: {refined_query}
        """)
    query_decomposition_chain = (
        query_decomposition_prompt  
        | ChatOllama(model="mistral:instruct")  
        | JsonOutputParser() 
    )
    decomposed_queries = query_decomposition_chain.invoke({"refined_query": refined_query})
    print("3️⃣ Query decomposition complete: ", decomposed_queries[:3], "...")  # Show first 3 queries
    return {"queries_for_retrieval": decomposed_queries}

def contextual_retrieval_node(state):
    seen_doc_ids = set()
    unique_docs = []
    
    for query in state['queries_for_retrieval']:
        results = retriever.retrieve_multi_modal(query, k=5, )
        for res in results:
            doc_id = res.get('doc_id') or res.get('summary_metadata', {}).get('doc_id')
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                unique_docs.append(res)
    print(f"4️⃣ Retrieved {len(unique_docs)} unique documents for {len(state['queries_for_retrieval'])} queries.")
    return {"retrieved_docs": unique_docs}

def relevancy_check_node(state):
    relevant_docs = []
    query = state.get('refined_query')
    docs = state.get('retrieved_docs')
    for doc in docs:
        modality = doc.get('modality') or (doc.get('original_metadata') or {}).get('type')
        summary = doc.get('summary', '')
        if modality in ('image', 'image_summary'):
            image_path = (doc.get('original_metadata') or {}).get('image_path')
            doc_desc = f"[IMAGE] Path: {image_path}\nSummary: {summary}"
        elif modality == 'table':
            doc_desc = f"[TABLE] Summary: {summary}"
        else:
            doc_desc = f"[TEXT] Summary: {summary}"
        prompt = (
            "You are a veterinary assistant AI. You are checking if a document is relevant and useful for answering a user's veterinary question. "
            "The document may be a summary of a textbook passage, a table, or an image (with a summary). "
            "Only say YES if the document contains information that would help answer the user's question, or provides context, steps, or background. "
            "If the document is off-topic, generic, or not helpful, say NO.\n\n"
            f"User query: {query}\nDocument: {doc_desc}\n\n"
            "Is this document relevant and useful for answering the query?"
            "Respond with only YES or NO."
        )
        messages = [{"role": "user", "content": prompt}]
        response = ollama.chat(
            model="mistral:instruct",
            messages=messages,
            options={"temperature": 0},
        )
        answer = response['message']['content'].strip().lower()
        if answer.startswith('yes'):
            relevant_docs.append(doc)
    
    print(f"5️⃣ Found {len(relevant_docs)} relevant documents out of {len(docs)} retrieved documents.")
    return {"relevant_docs": relevant_docs}

def extract_json_block(text):
    """Extracts the first {...} block from text, removing code fences and extra text."""
    # Remove code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
    if text.endswith("```"):
        text = re.sub(r"\s*```$", "", text)
    # Find the first '{' and last '}'
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1].strip()
    return text

def thinking_node(state):
    user_query = state.get("text_query", "")
    image_summary = state.get("image_summary", "")
    user_response = state.get("user_responses", {}).get("text", "")
    relevant_docs = state.get("relevant_docs")[0:6] # Limited due to context window size of Qwen3:8b
    followup_questions = state.get("followup_questions", [])

    if user_response:
        user_query += f"\nAdditional info from user: {user_response} for those questions that were asked: {followup_questions}"
    prompt = (
        "You are a veterinary assistant AI. The user is a pet owner with little veterinary knowledge. "
        "Explain in simple, actionable language, only suggesting home-care steps. If the case is serious, remind the user to see a vet. "
        "Base your answer strictly on the provided docs. "
        "If you need more info, specify what and which tool to use. "
        "You have access to a veterinary textbook and a database of documents, images, and tables. "
        "If the provided docs do not fully answer the user's question, you can suggest new search queries to retrieve more information. "
        "To do this, output 'Next action: retrieve more info' and provide a list of new queries that would help you find the answer. "
        "Example queries: [\"causes of cat ear bleeding\", \"cat ear infection symptoms\", \"treatment for dark wax in cat ear\"]\n\n"
        "When to use each action/tool:\n"
        "- Use 'retrieve more info' if the provided docs are insufficient, missing key details, or you are uncertain about the answer. Suggest specific, focused queries that would help you find the missing information in the textbook or database.\n"
        "- Use 'ask the user a question' if you need clarification, more details about the pet's symptoms, or additional context from the user to proceed.\n"
        "- Use 'ready to answer' if you have enough information from the provided docs to give a helpful, actionable answer.\n"
        "If you are missing key details, or the docs are insufficient, do not guess—ask for more info or suggest retrieval.\n\n"
        "Respond only in a clean JSON, using one of these formats:\n"
        '{\n'
        '  \"thinking\": \"your reasoning\",\n'
        '  \"next_action\": \"retrieve more info\",\n'
        '  \"queries\": [\"query1\", \"query2\"],\n'
        '}\n'
        "or\n"
        '{\n'
        '  \"thinking\": \"your reasoning\",\n'
        '  \"next_action\": \"ask the user a question\",\n'
        '  \"questions\": \"your answer\"\n'
        '}\n'
        "or\n"
        '{\n'
        '  \"thinking\": \"your reasoning\",\n'
        '  \"next_action\": \"ready to answer\",\n'
        '  \"answer\": \"your answer\",\n'
        '}\n\n'
        f"User question: {user_query}\n"
    )
    if image_summary: 
        prompt += f"Image summary: {image_summary}\n"
    prompt += "Relevant information from veterinary handbook:\n"
    for i, doc in enumerate(relevant_docs):
        modality = doc.get('modality') or (doc.get('original_metadata') or {}).get('type')
        doc_id = doc.get('doc_id') or (doc.get('original_metadata') or {}).get('doc_id')
        if modality in ('image', 'image_summary'):
            original_metadata = doc.get('original_metadata') or {}
            import json
            metadata_str = json.dumps(original_metadata, indent=2, ensure_ascii=False)
            prompt += (
                f"{i+1}. [IMAGE] The following is the original_metadata for this image (summary and image_path are included):\n"
                f"{metadata_str}\n(id: {doc_id})\n"
            )
        elif modality == 'table':
            summary = doc.get('summary', '')
            prompt += f"{i+1}. [TABLE] Summary: {summary} (id: {doc_id})\n"
        else:
            summary = doc.get('summary', '')
            prompt += f"{i+1}. [TEXT] Summary: {summary} (id: {doc_id})\n"

    messages = [{"role": "user", "content": prompt}]
    print("6️⃣ Thinking about the user's query...")
    print("Prompt length: ", len(prompt),)
    response = ollama.chat(
        model="qwen3:8b", 
        messages=messages,
        options={"temperature": 0.2},
    )
    llm_output = response['message']['content']
    think_match = re.search(r"<think>(.*?)</think>", llm_output, re.DOTALL | re.IGNORECASE)
    if think_match:
        reasoning = think_match.group(1).strip()
    else:
        reasoning = ""
    output_for_user = re.sub(r"<think>.*?</think>", "", llm_output, flags=re.DOTALL | re.IGNORECASE).strip()
    output_for_user_clean = extract_json_block(output_for_user)
    state["intermediate_thoughts"] = state.get("intermediate_thoughts", []) + [reasoning]


    if output_for_user_clean.startswith("{") and output_for_user_clean.endswith("}"):
        try:
            parsed = json.loads(output_for_user_clean)
            next_action = parsed.get("next_action", "").lower()
            if next_action == "retrieve more info":
                queries = parsed.get("queries", [])
                return {"next_action": "retrieve_more_info", "queries_for_retrieval": queries}
            elif "ask the user" in next_action or "ask user" in next_action:
                questions = parsed.get("questions", [])
                if isinstance(questions, str):
                    questions = [questions]
                elif not isinstance(questions, list):
                    questions = []
                return {"next_action": "ask_user", "followup_questions": questions}
            elif next_action == "ready to answer":
                final_answer = parsed.get("answer")
                return {"next_action": "final_answer_node", "final_answer": final_answer}
        except Exception as e:
            print("⚠️ Failed to parse JSON:", e)
    # fallback
    print("6️⃣ Now at the final return of thinking_node")
    return {"next_action": "final_answer_node", "final_answer": output_for_user_clean}

def retrieval_tool(state):
    queries = state.get("queries_for_retrieval", [])
    print("Retrieving more information for queries: ", queries)
    if not retriever or not queries:
        return state
    existing_docs = state.get("retrieved_docs", [])
    seen_doc_ids = set(
        doc.get('doc_id') or doc.get('summary_metadata', {}).get('doc_id')
        for doc in existing_docs
    )
    new_docs = []
    for query in queries:
        results = retriever.retrieve_multi_modal(query, k=5)
        for doc in results:
            doc_id = doc.get('doc_id') or doc.get('summary_metadata', {}).get('doc_id')
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                new_docs.append(doc)
    all_docs = existing_docs + new_docs
    state["retrieved_docs"] = all_docs
    state["next_action"] = None
    return {"next_action": None, "retrieved_docs": all_docs}

def ask_user_tool(state):
    followup_questions = state.get("followup_questions")
    print("❓ ",followup_questions)
    user_text = input("Your answer (press Enter to skip): ").strip()
    user_image_path = input("If you want to provide an image, enter the file path (or press Enter to skip): ").strip()
    user_response_entry = {}
    if user_text:
        user_response_entry["text"] = user_text
    if user_image_path:
        try:
            image_summary = describe_user_image(user_image_path) 
        except Exception as e:
            print(f"Error interpreting image: {e}")
            image_summary = None
        user_response_entry["image_path"] = user_image_path
        user_response_entry["image_summary"] = image_summary
    
    return { "next_action": None, "user_responses": user_response_entry}

def describe_user_image(image_path):
    prompt = (
        "You are a veterinary assistant AI. "
        "Describe the key features, symptoms, or findings in this image as they relate to a cat's health. "
        "Be concise and factual. If the image is unclear or irrelevant, say so."
    )
    messages = [{
        "role": "user",
        "content": prompt,
        "images": [image_path]
    }]
    response = ollama.chat(
        model="minicpm-v:8b",
        messages=messages,
        options={"temperature": 0.2}
    )
    return response['message']['content']

def final_answer_node(state):
    final_answer = state["final_answer"]
    CoT = state.get("intermediate_thoughts", [])
    print(" 🧠 Chain of Thoughts: ", CoT)

    print("7️⃣ Final answer generated: ", final_answer)

def build_graph():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import StateGraph, START
    checkpointer = InMemorySaver()
    builder = StateGraph(GraphState)  

    builder.add_node("query_handler", query_handler)
    builder.add_node("query_refinement", query_refinement_node)
    builder.add_node("query_decomposition", query_decomposition)
    builder.add_node("contextual_retrieval", contextual_retrieval_node)
    builder.add_node("relevancy_check", relevancy_check_node)
    builder.add_node("thinking", thinking_node)
    builder.add_node("retrieval_tool", retrieval_tool)
    builder.add_node("ask_user_tool", ask_user_tool)
    builder.add_node("final_answer_node", final_answer_node)

    builder.add_edge(START, "query_handler")
    def query_type_router(state):
        query_type = state.get("query_type")
        if query_type == "irrelevant":
            return "query_handler"  # Loop back for irrelevant queries
        else:
            return "query_refinement"  # Proceed for Q&A or emergency

    builder.add_conditional_edges(
        "query_handler",
        query_type_router,
        {
            "query_handler": "query_handler",
            "query_refinement": "query_refinement"
        }
    )

    builder.add_edge("query_refinement", "query_decomposition")
    builder.add_edge("query_decomposition", "contextual_retrieval")
    builder.add_edge("contextual_retrieval","relevancy_check")
    builder.add_edge("relevancy_check", "thinking")

    def thinking_router(state):
        action = state.get("next_action")
        if action == "retrieve_more_info":
            return "retrieval_tool"
        elif action == "ask_user":
            return "ask_user_tool"
        elif action == "ready_to_answer":
            return "final_answer_node"
        else:
            return "final_answer_node"
    builder.add_conditional_edges(
        "thinking",
        thinking_router,
        {
            "retrieval_tool": "retrieval_tool",
            "ask_user_tool": "ask_user_tool",
            "final_answer_node": "final_answer_node"
        }
    )
    builder.add_edge("retrieval_tool", "contextual_retrieval")
    builder.add_edge("ask_user_tool", "thinking")
    graph = builder.compile(checkpointer=checkpointer)
    #graph = builder.compile()

    return graph

def init_retriever():
    persist_directory = '../../chroma/Cat_Owners_Home_Veterinary_Handbook'
    id_key = "doc_id"
    text_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    open_clip_embeddings = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")
    text_vectorstore = Chroma(
        collection_name="text_summaries_and_tables_and_image_summaries",
        embedding_function=text_embeddings,
        persist_directory=persist_directory
    )
    text_docstore = Chroma(
        collection_name="text_originals",
        embedding_function=text_embeddings,
        persist_directory=persist_directory
    )
    image_vectorstore = Chroma(
        collection_name="images",
        embedding_function=open_clip_embeddings,
        persist_directory=persist_directory
    )
    image_docstore = Chroma(
        collection_name="image_originals",
        embedding_function=open_clip_embeddings,
        persist_directory=persist_directory
    )
    retriever = UnifiedRetriever(
        text_vectorstore, text_docstore, image_vectorstore, image_docstore, id_key=id_key
    )
    return retriever

def main():
    global retriever
    retriever = init_retriever() 
    graph = build_graph()
    print("Welcome to the Veterinary Assistant!")
    text_query = input("Enter your question about your cat (or pet): ").strip()
    image_path = input("If you want to provide an image, enter the file path (or press Enter to skip): ").strip()
    initial_state = {
        "text_query": text_query,
        "image_path": image_path if image_path else None,
        "loop_count": 0,
         
    }
    result = graph.invoke(initial_state,  {"configurable": {"thread_id": str(uuid.uuid4())}})
    print("Session complete.")

if __name__ == "__main__":
    main()