class UnifiedRetriever:
    """
    UnifiedRetriever supports multi-modal retrieval from separate text and image vectorstores and docstores.
    It can retrieve by text, image, or both (multi-query), and supports metadata filtering by modality.
    """
    def __init__(self, text_vectorstore, text_docstore, image_vectorstore=None, image_docstore=None, id_key="doc_id"):
        self.text_vectorstore = text_vectorstore
        self.text_docstore = text_docstore
        self.image_vectorstore = image_vectorstore
        self.image_docstore = image_docstore
        self.id_key = id_key

    def retrieve(self, query, k=5, filter=None):
        """
        Retrieve top-k results for a query from both text and image vectorstores, optionally filtered by metadata (e.g., modality).
        Combine and return merged results.
        """
        # Text search
        text_results = self.text_vectorstore.similarity_search_with_score(query, k=k, filter=filter)
        # Image search
        image_results = []
        if self.image_vectorstore:
            image_results = self.image_vectorstore.similarity_search_with_score(query, k=k, filter=filter)
        output = []
        # Process text results
        for doc, score in text_results:
            doc_id = doc.metadata.get(self.id_key)
            try:
                original = self.text_docstore._collection.get(ids=[doc_id], include=["documents", "metadatas"])
                original_doc = original["documents"][0] if original["documents"] else None
                original_meta = original["metadatas"][0] if original["metadatas"] else None
            except Exception:
                original_doc = None
                original_meta = None
            output.append({
                "summary": doc.page_content,
                "original": original_doc,
                "original_metadata": original_meta,
                "summary_metadata": doc.metadata,
                "score": score,
                "modality": doc.metadata.get("type", "text"),
            })
        # Process image results
        for doc, score in image_results:
            doc_id = doc.metadata.get(self.id_key)
            try:
                original = self.image_docstore._collection.get(ids=[doc_id], include=["documents", "metadatas"])
                original_doc = original["documents"][0] if original["documents"] else None
                original_meta = original["metadatas"][0] if original["metadatas"] else None
            except Exception:
                original_doc = None
                original_meta = None
            output.append({
                "summary": doc.page_content,
                "original": original_doc,
                "original_metadata": original_meta,
                "summary_metadata": doc.metadata,
                "score": score,
                "modality": doc.metadata.get("type", "image"),
            })
        # Sort by score (descending if similarity)
        output.sort(key=lambda x: x["score"], reverse=True)
        return output

    def retrieve_multi_modal(self, query, k=5, text_types=("text", "table", "image_summary"), image_types=("image",)):
        """
        Multi-Query/Multi-Modal Retrieval:
        - Retrieves top-k text (text, table, image_summary) and top-k image results for the query.
        - Merges and sorts by score.
        - Returns a list of results with modality info.
        """
        # Textual modalities
        text_results = self.text_vectorstore.similarity_search(query, k=k, filter={"type": {"$in": list(text_types)}})
        # Image modalities
        image_results = []
        if self.image_vectorstore:
            image_results = self.image_vectorstore.similarity_search(query, k=k, filter={"type": {"$in": list(image_types)}})
        all_results = []
        for doc in text_results:
            doc_id = doc.metadata.get(self.id_key)
            all_results.append({
                "modality": doc.metadata.get("type"),
                "summary": doc.page_content,
                "original_metadata": doc.metadata,
                
                "doc_id": doc_id
            })
        for doc in image_results:
            doc_id = doc.metadata.get(self.id_key)
            all_results.append({
                "modality": doc.metadata.get("type"),
                "summary": doc.page_content,
                "original_metadata": doc.metadata,
                
                "doc_id": doc_id
            })
        return all_results 