import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import gradio as gr

# 1. Load Resources
class LegalAI:
    def __init__(self):
        # Load Vector Store
        self.index = faiss.read_index("lexfind_embeddings.npy")
        with open("lexfind_documents.txt") as f:
            self.docs = [line.strip() for line in f.read().split("---") if line.strip()]
        
        # Load Embedding Model
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Load Mistral-7B
        self.llm = Llama(
            model_path="/data/models_weights/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf",
            n_ctx=4096,  # Context window
            n_gpu_layers=40  # Use GPU layers (adjust based on VRAM)
        )

    def retrieve_relevant_laws(self, query, top_k=3):
        # Vector Search
        query_embed = self.embedder.encode([query])
        distances, indices = self.index.search(query_embed, top_k)
        return [self.docs[i] for i in indices[0]]

    def generate_response(self, query):
        # Step 1: Retrieve relevant laws
        laws = self.retrieve_relevant_laws(query)
        context = "\n\n".join(laws)
        
        # Step 2: Generate answer with Mistral
        prompt = f"""<s>[INST] You are a Swiss Legal AI Assistant. 
        Use this context to answer the user's question:
        {context}
        
        Question: {query} 
        Answer concisely in 2-3 sentences. [/INST]"""
        
        output = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.3  # Lower = more deterministic
        )
        return output['choices'][0]['text']

# 2. Gradio Interface
def chat_interface(query, history):
    ai = LegalAI()
    response = ai.generate_response(query)
    return response

demo = gr.ChatInterface(
    fn=chat_interface,
    title="Swiss Legal AI Assistant",
    description="Ask questions about Swiss law"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
