import faiss
import numpy as np

# Load your embeddings
embeddings = np.load("legal_embeddings_SAMPLE.npy")  # Shape: (num_docs, embedding_dim)

# Initialize FAISS index
dimension = embeddings.shape[1]  # e.g., 384 for all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)  # L2 distance metric

# Add embeddings to the index
index.add(embeddings)

# Save the index
faiss.write_index(index, "legal_docs_faiss_index.index")
print("? FAISS index created with", index.ntotal, "documents")
