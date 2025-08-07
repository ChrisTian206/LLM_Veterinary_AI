import sys
import os
import uuid
import requests
import json
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))
import ollama
import re

# Load environment variables
load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Optional, List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_chroma import Chroma
from textbook_to_db.unified_retriever import UnifiedRetriever
from tavily import TavilyClient
from langchain_chroma import Chroma
from textbook_to_db.unified_retriever import UnifiedRetriever
from tavily import TavilyClient


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
    ai_response: Optional[List[str]]
    path_taken: Optional[List[str]]
    error: Optional[str]
    latest_user_input: Optional[str]
    retrieval_count: Optional[int]
    tavily_query: Optional[str]
    web_search_results_log: Optional[List[Dict[str, Any]]]
    new_docs_count: Optional[int]
    action_history: Optional[List[str]]
    
def query_handler(state):
    text_query = state.get("text_query", "")
    image_path = state.get("image_path", None)
    prompt = (
        "You are a domain classifier for a feline veterinary assistant. "
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
        model="mistral:instruct",
        messages=messages,
        options={"temperature": 0.2}
    )
    result = response['message']['content'].strip().lower()
    if result not in ['irrelevant', 'emergency', 'q&a']:
        result = 'irrelevant'

    print(f"1️⃣ Query type determined: {result}")
    return {"query_type": result}

def query_handler_tool(state):
    original_query = state.get("text_query", "")

    prompt = (
        "You are a friendly and professional feline veterinary assistant. "
        "A user just asked the following question:\n"
        f"\"{original_query}\"\n\n"
        "This query appears to be unrelated to our feline veterinary or animal health topics. "
        "Kindly inform the user that their question isn't within scope and ask them to provide a veterinary-related query. "
        "Suggest a few example questions related to cats health, animal care, or cat veterinary emergencies.\n\n"
        "Respond in a warm and helpful tone, but don't answer the original question."
    )

    response = ollama.chat(
        model="mistral:instruct",  # or whatever model you're using
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7}
    )

    assistant_reply = response["message"]["content"]
    print(f"\n🤖 {assistant_reply}\n")

    # Ask user again
    new_text = input("📝 Please enter your new question: ")
    new_image = input("📷 Optional: Enter image path (or press Enter to skip): ").strip()

    return {
        "text_query": new_text,
        "image_path": new_image or None,
    }

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
    relevant_docs = state.get("relevant_docs", [])[:10]
    action_history = state.get("action_history", [])
    latest_user_input = state.get("latest_user_input", "")
    
    consecutive_retrievals = 0
    for action in reversed(action_history):
        if action == "retrieve":
            consecutive_retrievals += 1
        else:
            break

    prompt = (
        "You are a veterinary assistant AI helping pet owners with at-home vet care. The user is a pet owner with little veterinary knowledge. "
        "You are providing a at-home care advice based on book 'Cat Owners' Home Veterinary Handbook by Debra M. Eldredge'. This entire book and trusted web sources are available to you, and you can retrieve information from the book using the retrieve tool and search trusted veterinary websites using the web search tool."

        "Your job is to help the user by understanding their needs, asking follow-up questions, retrieving relevant information, and providing a comprehensive answer based on the user's query, the retrieved documents, and web search results so that your answer is safe, actionable, and reliable. In the retrieved documents, there are also images. \n\n"

        "Do not refer to internal document numbers or sources such as “document 2” or “document 10” in your response. Instead, explain the guidance directly as if you are speaking to the pet owner. However, if you think those images are helpful to support your answer, you can include the image path, which can be found in its original_metadata, in your answer. "

        "Respond only in a clean JSON, following the format in provided examples.\n"
        "Base your answer strictly on the provided docs. "
        "If you need more info, specify what and which tool to use. "
        
        "You should be thorough and make multiple retrieval calls or wikipedia search to gather comprehensive information before answering."
        "Consider making more targeted retrieval calls if the information seems incomplete for the specific symptoms described.\n"
    )
    
    if consecutive_retrievals >= 2:
            prompt += (
                f"\n⚠️ **IMPORTANT**: You have made {consecutive_retrievals} consecutive retrieval calls. "
                "Consider using web search to get broader context, current medical information, or cross-reference information from trusted veterinary sources "
                "before making another retrieval call. Web search can provide valuable supplementary and up-to-date information.\n"
            )
    
    prompt += (
        "Here are the tools you can use:\n"

        "- **retrieve more info**: If you need more specific information to give a safe, actionable answer, use this tool to search the veterinary handbook or database for additional details. Focus on specific symptoms or conditions mentioned by the user. Here is an example:\n"
        "Example queries: [\"cat urinary blockage symptoms\", \"cat difficulty sitting spinal issues\", \"feline urethral obstruction emergency\"]\n\n"
        "Output the next action and queries in this format:\n"
        '{\n'
        '  \"thinking\": \"your reasoning\",\n'
        '  \"next_action\": \"retrieve more info\",\n'
        '  \"queries\": [\"query1\", \"query2\"],\n'
        '}\n'

        "- **search web**: If you need current information, general background, or want to cross-reference veterinary conditions from trusted sources, use this tool to search the web using Tavily. Here is an example:\n"
        "Example query: \"feline urinary tract infection symptoms treatment\"\n\n"
        "Output the next action and query in this format:\n"
        '{\n'
        '  \"thinking\": \"your reasoning\",\n'
        '  \"next_action\": \"search web\",\n'
        '  \"tavily_query\": \"your search query\"\n'
        '}\n'

        "- **continue conversation**: If you want to ask the user NEW follow-up questions (not ones already asked) or ask the user to take a picture or provide information and continue the conversation, use this tool. Here is an example:\n"
        "Example response: \"Based on the symptoms you described, this could be a urinary blockage. Can you tell me if your cat is straining to urinate?\"\n\n"
        '{\n'
        '  \"thinking\": \"your reasoning\",\n'
        '  \"next_action\": \"continue conversation\",\n'
        '  \"response\": \"your response or question to the user\",\n'
        '}\n'

    )

    prompt += f"Original user query: {user_query}\n"
    if latest_user_input:
        prompt += f"Latest user input: {latest_user_input}\n"

    if image_summary: 
        prompt += f"Image summary: {image_summary}\n"
    
    prompt += f"\nHere are {len(relevant_docs)} relevant documents from veterinary handbook:\n"
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

    # Add web search results if available
    web_search_results = state.get("web_search_results", [])
    web_search_log = state.get("web_search_results_log", [])
    
    if web_search_results:
        prompt += "\nHere are relevant web search results:\n"
        for i, result in enumerate(web_search_results):
            prompt += f"{i+1}. [WEB] Title: {result.get('title', '')}\nURL: {result.get('url', '')}\nContent: {result.get('content', '')[:500]}...\nScore: {result.get('score', 0)}\n"
    
    # Add the most recent web search log entry if available
    if web_search_log:
        latest_search = web_search_log[-1]
        prompt += f"\n📊 Latest Web Search Summary:\n"
        prompt += f"Query: {latest_search.get('query', '')}\n"
        prompt += f"Found {latest_search.get('results_count', 0)} results\n"
        prompt += f"Top sources: "
        top_sources = [result.get('url', 'unknown') for result in latest_search.get('results', [])[:2]]
        prompt += f"{', '.join(top_sources)}\n"
        prompt += "\n**IMPORTANT**: When providing your response, make sure to reference these web search sources for credibility. Mention the specific websites or sources you found this information from.\n"

    # Add conversation history only if not a new topic
    q_list = state.get("ai_response", [])
    a_list = state.get("user_responses", [])
    if q_list and a_list and len(a_list) > 0:
        recent_pairs = list(zip(q_list, a_list)) 
        if recent_pairs:
            prompt += "\nRecent conversation history (DO NOT repeat these questions):\n"
            for q, a in recent_pairs:
                prompt += f"AI : {q}\n User: {a}\n"
                

    messages = [{"role": "user", "content": prompt}]
    print("6️⃣ Thinking about the user's query...")
    print("Prompt length: ", len(prompt))
    
    response = ollama.chat(
        model="qwen3:8b", 
        messages=messages,
        options={"temperature": 0.1},
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
            formatted_json = json.dumps(parsed, indent=2, ensure_ascii=False)
            print("🧠: ")
            print(f"\033[90m{formatted_json}\033[0m")  # cyan color
            next_action = parsed.get("next_action", "").lower()
            updated_action_history = action_history.copy()

            
            if next_action == "retrieve more info":
                updated_action_history.append("retrieve")
                queries = parsed.get("queries", [])
                print(f"📖 Checking Books for: {queries}")
                return {"next_action": "retrieve_more_info", "queries_for_retrieval": queries, "action_history": updated_action_history}
            elif next_action == "search web":
                tavily_query = parsed.get("tavily_query", "")
                updated_action_history.append("web_search")
                print(f"🌐 Searching Web for: {tavily_query}")
                return {"next_action": "search_web", "tavily_query": tavily_query, "action_history": updated_action_history}
            elif "continue conversation" in next_action:
                response_text = parsed.get("response", "")
                updated_action_history.append("continue_conversation")
                ai_response = state.get("ai_response", [])
                ai_response.append(response_text)
                return {"next_action": "user_interaction", "ai_response": ai_response, "action_history": updated_action_history}
                    
        
        except Exception as e:
            print("⚠️ Failed to parse JSON:", e)
            print(f"\033[90m💥 DEBUG: Raw LLM output: {llm_output}\033[0m")
            print(f"\033[90m💥 DEBUG: Cleaned output: {output_for_user_clean}\033[0m")

    else:
        print("⚠️ LLM output is not valid JSON. Returning to user interaction.")
        print(f"\033[90m💥 DEBUG: Raw LLM output: {llm_output}\033[0m")
        print(f"\033[90m💥 DEBUG: Cleaned output: {output_for_user_clean}\033[0m")
        sys.exit("LLM output is not valid JSON. Exiting...")

def user_interaction_node(state):
    """Node for continuous user interaction - handles questions and responses"""
    # print("7️⃣ User interaction nodecx activated.")
    ai_response = state.get("ai_response", [])
    if ai_response and len(ai_response) > 0:
        print(f"\033[92m🤖 {ai_response[-1]}\033[0m")  # Green

    
    # Get user input
    user_input = input("Your response (or type '/bye' to exit): ").strip()
    
    # Check for exit command
    if user_input.lower() == '/bye':
        print("\nThank you for using the Veterinary Assistant! Take care of your pet! 🐱")
        return {"next_action": "exit", "user_input": user_input}
    
    # Check if user wants to provide an image
    image_path = input("If you want to provide an image, enter the file path (or press Enter to skip): ").strip()
    
    # Update state with user response
    existing_responses = state.get("user_responses", [])
    existing_responses.append(user_input)
    
    # Clear followup questions since user has responded
    result = {
        "next_action": "continue_conversation",
        "user_responses": existing_responses,
        "latest_user_input": user_input,
    }
    
    # If user provided an image, get summary and add to state
    if image_path and os.path.exists(image_path):
        print("📸 Processing your image...")
        image_summary = get_image_summary(image_path)
        result["image_summary"] = image_summary
        result["image_path"] = image_path
    
    return result

def retrieval_tool(state):
    queries = state.get("queries_for_retrieval", [])
    print("🔧 Retrieving more information for queries: ", queries)
    if not retriever or not queries:
        return {"next_action": "process_new_docs"}
    
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
    
    # Increment retrieval count
    retrieval_count = state.get("retrieval_count", 0) + 1
    
    print(f"📊 Added {len(new_docs)} new documents. Total: {len(all_docs)} documents.")
    
    return {
        "next_action": "process_new_docs", 
        "retrieved_docs": all_docs, 
        "retrieval_count": retrieval_count,
        "new_docs_count": len(new_docs)
    }

def process_new_docs_node(state):
    """Process newly retrieved documents through relevancy check"""
    all_docs = state.get("retrieved_docs", [])
    existing_relevant = state.get("relevant_docs", [])
    new_docs_count = state.get("new_docs_count", 0)
    
    if new_docs_count == 0:
        return {"next_action": None}
    
    # Get only the newly added documents
    new_docs = all_docs[-new_docs_count:] if new_docs_count > 0 else []
    
    if not new_docs:
        return {"next_action": None}
    
    print(f"🔍 Processing {len(new_docs)} newly retrieved documents for relevancy...")
    
    # Run relevancy check on new documents
    relevant_new_docs = []
    query = state.get('refined_query', '') or state.get('text_query', '')
    latest_user_input = state.get('latest_user_input', '')
    
    # Use latest user input if available for more context
    check_query = f"{query}. Latest user input: {latest_user_input}" if latest_user_input else query
    
    for doc in new_docs:
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
            f"User query and context: {check_query}\nDocument: {doc_desc}\n\n"
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
            relevant_new_docs.append(doc)
    
    # Combine with existing relevant docs, avoiding duplicates
    existing_relevant_ids = set(
        doc.get('doc_id') or doc.get('summary_metadata', {}).get('doc_id')
        for doc in existing_relevant
    )
    
    combined_relevant = existing_relevant.copy()
    for doc in relevant_new_docs:
        doc_id = doc.get('doc_id') or doc.get('summary_metadata', {}).get('doc_id')
        if doc_id not in existing_relevant_ids:
            combined_relevant.append(doc)
    
    print(f"✅ Found {len(relevant_new_docs)} relevant documents from new retrieval. Total relevant: {len(combined_relevant)}")
    
    return {
        "next_action": None,
        "relevant_docs": combined_relevant
    }

def tavily_search_tool(state):
    """Search web using Tavily for veterinary information"""
    query = state.get("tavily_query", "")
    
    if not query:
        return {"next_action": None}
    
    try:
        # Initialize Tavily client using environment variable
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            print("⚠️ TAVILY_API_KEY not found in environment variables")
            return {"next_action": None}
        
        tavily_client = TavilyClient(api_key=api_key)
        
        # Enhanced query for veterinary context
        veterinary_query = f"veterinary {query} cats feline health"
        
        print(f"🌐 Searching web for: {veterinary_query}")
        
        # Search using Tavily
        search_results = tavily_client.search(
            query=veterinary_query,
            search_depth="advanced",
            max_results=5,
            include_domains=[
                "https://www.avma.org/",
                "https://www.merckvetmanual.com/",
                "https://www.vet.cornell.edu/",
                "https://www.vetmed.ucdavis.edu/",
                "https://vcahospitals.com/know-your-pet",
                "https://www.petmd.com/",
                "https://www.aspca.org/pet-care",
                "https://www.petpoisonhelpline.com/",
                "https://www.wormsandgermsblog.com/",
                "https://www.vin.com/"
                ],
            exclude_domains=["reddit.com", "quora.com","wikipedia.org"],
            include_answer=True,
            include_raw_content=True
        )
        
        if search_results and search_results.get("results"):
            # Process multiple results
            processed_results = []
            
            for result in search_results["results"][:3]:  # Take top 3 results
                processed_result = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0),
                    "query": query
                }
                processed_results.append(processed_result)
            
            # Add Tavily's direct answer if available
            if search_results.get("answer"):
                direct_answer = {
                    "title": f"Direct Answer: {query}",
                    "url": "tavily_direct_answer",
                    "content": search_results["answer"],
                    "score": 1.0,
                    "query": query
                }
                processed_results.insert(0, direct_answer)  # Put direct answer first
            
            existing_results = state.get("web_search_results", [])
            existing_results.extend(processed_results)
            
            # Create a log entry for this search
            log_entry = {
                "query": query,
                "veterinary_query": veterinary_query,
                "timestamp": str(uuid.uuid4())[:8],  # Simple timestamp-like ID
                "results_count": len(processed_results),
                "results": processed_results
            }
            
            # Add to the log
            existing_log = state.get("web_search_results_log", [])
            existing_log.append(log_entry)
            
            print(f"📚 Found {len(processed_results)} web search results for: {query}")
            return {
                "next_action": None, 
                "web_search_results": existing_results,
                "web_search_results_log": existing_log
            }
        else:
            print(f"⚠️ No web search results found for: {query}")
            return {"next_action": None}
            
    except Exception as e:
        print(f"⚠️ Error searching web: {e}")
        return {"next_action": None}

def final_answer_node(state):
    final_answer = state["final_answer"]
    print(f"🤖 {final_answer}")
    return {"next_action": "user_interaction"}

def build_graph():
    checkpointer = InMemorySaver()
    builder = StateGraph(GraphState)  

    builder.add_node("query_handler", query_handler)
    builder.add_node("query_handler_tool", query_handler_tool)
    builder.add_node("query_refinement", query_refinement_node)
    builder.add_node("query_decomposition", query_decomposition)
    builder.add_node("contextual_retrieval", contextual_retrieval_node)
    builder.add_node("relevancy_check", relevancy_check_node)
    builder.add_node("thinking", thinking_node)
    builder.add_node("retrieval_tool", retrieval_tool)
    builder.add_node("process_new_docs_node", process_new_docs_node)  # New node
    builder.add_node("tavily_search_tool", tavily_search_tool)
    builder.add_node("user_interaction_node", user_interaction_node)
    builder.add_node("final_answer_node", final_answer_node)

    builder.add_edge(START, "query_handler")

    def query_type_router(state):
        query_type = state.get("query_type")
        if query_type == "irrelevant":
            return "query_handler_tool"  
        else:
            return "query_refinement"  # Proceed for Q&A or emergency

    builder.add_conditional_edges(
        "query_handler",
        query_type_router,
        {
            "query_handler_tool": "query_handler_tool",
            "query_refinement": "query_refinement"
        }
    )

    builder.add_edge("query_handler_tool", "query_handler")
    builder.add_edge("query_refinement", "query_decomposition")
    builder.add_edge("query_decomposition", "contextual_retrieval")
    builder.add_edge("contextual_retrieval","relevancy_check")
    builder.add_edge("relevancy_check", "thinking")

    def thinking_router(state):
        action = state.get("next_action")
        if action == "retrieve_more_info":
            return "retrieval_tool"
        elif action == "search_web":
            return "tavily_search_tool"
        elif action == "user_interaction":
            return "user_interaction_node"
        else:
            return "user_interaction_node"
    
    builder.add_conditional_edges(
        "thinking",
        thinking_router,
        {
            "retrieval_tool": "retrieval_tool",
            "tavily_search_tool": "tavily_search_tool",
            "user_interaction_node": "user_interaction_node",
        }
    )
    
    def user_interaction_router(state):
        action = state.get("next_action")
        if action == "exit":
            return "END"
        else:
            return "thinking"
    
    builder.add_conditional_edges(
        "user_interaction_node",
        user_interaction_router,
        {
            "thinking": "thinking",
            "END": END  
        }
    )
    
    def retrieval_router(state):
        action = state.get("next_action")
        if action == "process_new_docs":
            return "process_new_docs_node"
        else:
            return "thinking"
    
    builder.add_conditional_edges(
        "retrieval_tool",
        retrieval_router,
        {
            "process_new_docs_node": "process_new_docs_node",
            "thinking": "thinking"
        }
    )
    
    builder.add_edge("process_new_docs_node", "thinking")
    builder.add_edge("tavily_search_tool", "thinking")
    builder.add_edge("final_answer_node", "user_interaction_node")
    
    graph = builder.compile(checkpointer=checkpointer)
    
    # print("***************Graph built successfully! 🐾****************")
    # from IPython.display import Image, display
    # graph_image = graph.get_graph(xray=True).draw_mermaid_png()
    # with open("graph_visualization.png", "wb") as f:
    #     f.write(graph_image)
    # print("Graph saved as graph_visualization.png")


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
    print("You can have a continuous conversation with the AI.")
    print("Type '/bye' at any time to exit the conversation.\n")
    
    thread_id = str(uuid.uuid4())
    
    text_query = input("Enter your question about your cat (or pet): ").strip()
    
    # Check for exit command
    if text_query.lower() == '/bye':
        return
    
    image_path = input("If you want to provide an image, enter the file path (or press Enter to skip): ").strip()
    
    initial_state = {
        "text_query": text_query,
        "image_path": image_path if image_path else None,
        "loop_count": 0,
        "retrieval_count": 0,
        "web_search_results": [],
        "web_search_results_log": [],
        "user_responses": [],
    }
    
    try:
        # Start the conversation
        result = graph.invoke(initial_state, {"configurable": {"thread_id": thread_id, "recursion_limit": 100}})
    except Exception as e:
        print(f"⚠️ An error occurred: {e}")
        print("The conversation has ended.")

if __name__ == "__main__":
    main()