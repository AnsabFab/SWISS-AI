import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load the FAISS index and embedding model
index = faiss.read_index("legal_docs_faiss_index.index")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Load your actual documents (replace with your data loading logic)
def load_document_texts():
    """Load the text of documents corresponding to the embeddings"""
    # This should match how you created embeddings earlier
    # Example: Read from sample_documents.txt or your original PDFs
    with open("sample_documents.txt") as f:
        return [doc.strip() for doc in f.read().split("\n---\n") if doc.strip()]
    
documents = load_document_texts()
print(f"Loaded {len(documents)} documents")

# 3. Robust semantic search function
def semantic_search(query, top_k=3):
    try:
        # Verify we have documents
        if not documents:
            raise ValueError("No documents loaded!")
        
        # Encode query
        query_embedding = model.encode([query])
        
        # Search FAISS
        distances, indices = index.search(query_embedding, top_k)
        
        # Verify results exist
        if indices.shape[1] == 0:
            print("No results found!")
            return
            
        print(f"\n?? Query: '{query}'\n")
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            # Validate index range
            if idx >= len(documents):
                print(f"?? Warning: Index {idx} out of range (max {len(documents)-1})")
                continue
                
            print(f"?? Result #{i+1} (Similarity: {1-dist:.2f}):")
            print(documents[idx][:200] + "...\n")
            
    except Exception as e:
        print(f"Error during search: {str(e)}")

# 4. Example queries
semantic_search("What are the key provisions of Swiss data protection laws?")
semantic_search("How are contracts enforced in Switzerland?")
