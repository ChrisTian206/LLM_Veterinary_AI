class UnifiedRetriever:
    """
    UnifiedRetriever supports multi-modal retrieval from separate text and image vectorstores and docstores.
    It can retrieve by text, image, or both (multi-query), and supports metadata filtering by modality.
    
    Supports dual-collection mode where text/tables use one collection and images use another.
    """
    def __init__(self, text_vectorstore, text_docstore, image_vectorstore=None, image_docstore=None, id_key="doc_id",
                 text_table_vectorstore=None, text_table_docstore=None):
        # Primary collection (can be used for images or fallback)
        self.text_vectorstore = text_vectorstore
        self.text_docstore = text_docstore
        
        # Optional separate collections for text/tables (better accuracy)
        self.text_table_vectorstore = text_table_vectorstore if text_table_vectorstore else text_vectorstore
        self.text_table_docstore = text_table_docstore if text_table_docstore else text_docstore
        
        # Image collections
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
        - Uses separate collections: text_table_vectorstore for text/tables, text_vectorstore for images
        - Merges and sorts by score.
        - Returns a list of results with modality info.
        """
        all_results = []
        
        # Textual modalities (text, table, image_summary) - use dedicated text/table collection
        if text_types:
            text_results = self.text_table_vectorstore.similarity_search(
                query, k=k, filter={"type": {"$in": list(text_types)}}
            )
            for doc in text_results:
                doc_id = doc.metadata.get(self.id_key)
                all_results.append({
                    "modality": doc.metadata.get("type"),
                    "summary": doc.page_content,
                    "original_metadata": doc.metadata,
                    "doc_id": doc_id,
                    "docstore": self.text_table_docstore  # Track which docstore to use
                })
        
        # Image modalities - use original collection with images
        if image_types and self.image_vectorstore:
            image_results = self.image_vectorstore.similarity_search(
                query, k=k, filter={"type": {"$in": list(image_types)}}
            )
            for doc in image_results:
                doc_id = doc.metadata.get(self.id_key)
                all_results.append({
                    "modality": doc.metadata.get("type"),
                    "summary": doc.page_content,
                    "original_metadata": doc.metadata,
                    "doc_id": doc_id,
                    "docstore": self.image_docstore if self.image_docstore else self.text_docstore
                })
        elif image_types and not self.image_vectorstore:
            # Fallback: use text_vectorstore (original collection) for images
            image_results = self.text_vectorstore.similarity_search(
                query, k=k, filter={"type": {"$in": list(image_types)}}
            )
            for doc in image_results:
                doc_id = doc.metadata.get(self.id_key)
                all_results.append({
                    "modality": doc.metadata.get("type"),
                    "summary": doc.page_content,
                    "original_metadata": doc.metadata,
                    "doc_id": doc_id,
                    "docstore": self.text_docstore
                })
        
        return all_results
    
    def retrieve_multi_query_augmented(self, queries, k=5, text_types=("text", "table", "image_summary"), image_types=("image",)):
        """
        Multi-Query Augmented Retrieval:
        - Takes multiple query variations
        - Retrieves documents for each query
        - Deduplicates and merges results
        - Returns augmented list of unique documents
        
        Args:
            queries: List of query strings
            k: Number of results per query
            text_types: Modalities to retrieve from text/table collection
            image_types: Modalities to retrieve from image collection
            
        Returns:
            List of unique documents with their metadata
        """
        seen_doc_ids = set()
        all_unique_results = []
        
        for query in queries:
            results = self.retrieve_multi_modal(query, k=k, text_types=text_types, image_types=image_types)
            
            for doc in results:
                doc_id = doc.get("doc_id")
                # Only add if we haven't seen this document yet
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    all_unique_results.append(doc)
        
        return all_unique_results 