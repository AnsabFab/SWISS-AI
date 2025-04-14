import os
import numpy as np
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import gradio as gr

# 1. Configuration
MODEL_PATH = "/data/models_weights/phi-2"
DOCUMENTS_FILE = "/data/scripts/fedlex_documents.txt"  # Single text file containing all documents
EMBEDDER_NAME = "mixedbread-ai/mxbai-embed-large-v1"
FAISS_INDEX_PATH = "/data/scripts/legal_docs_faiss_index.index"  # Existing FAISS index

# 2. Document Loader (for the single TXT file)
class LegalDocumentLoader:
    def __init__(self):
        self.documents = []
        self.load_documents()
        
    def load_documents(self):
        print(f"Loading documents from {DOCUMENTS_FILE}...")
        try:
            with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split documents by the DOC separator
            doc_entries = content.split("---\n")
            
            for doc in doc_entries:
                if doc.strip():  # Only process non-empty documents
                    # Extract document number and content
                    parts = doc.split("\n", 2)
                    if len(parts) >= 3:
                        doc_num = parts[0].strip()
                        doc_title = parts[1].strip()
                        doc_content = parts[2].strip()
                        
                        self.documents.append({
                            "id": doc_num,
                            "title": doc_title,
                            "text": doc_content[:100000]  # Limit to first 100k chars
                        })
                        
        except Exception as e:
            print(f"?? Error loading documents: {str(e)}")
        print(f"Loaded {len(self.documents)} documents")

# 3. Document Retriever with existing FAISS index
class LegalRetriever:
    def __init__(self):
        self.loader = LegalDocumentLoader()
        self.embedder = SentenceTransformer(EMBEDDER_NAME)
        self.index = self.load_index()
        
    def load_index(self):
        if os.path.exists(FAISS_INDEX_PATH):
            print("Loading existing FAISS index")
            return faiss.read_index(FAISS_INDEX_PATH)
        else:
            raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
            
    def retrieve(self, query: str, top_k: int = 3):
        query_embed = self.embedder.encode([query])
        distances, indices = self.index.search(query_embed, top_k)
        return [self.loader.documents[i] for i in indices[0]]

# 4. Load Models
print("Loading Phi-2.5 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)

retriever = LegalRetriever()

# 5. Generation Function
def generate_response(query: str):
    try:
        # Retrieve context
        context_docs = retriever.retrieve(query)
        context = "\n".join([f"?? Document {doc['id']} - {doc['title']}:\n{doc['text'][:2000]}..." 
                           for doc in context_docs])
        
        # Generate response
        prompt = f"""Task: Answer this Swiss legal question using the context.
        
Context:
{context}

Question: {query}

Answer concisely with references to document numbers:"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.3
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Answer concisely with references to document numbers:")[-1].strip()
    
    except Exception as e:
        return f"? Error: {str(e)}"

# 6. Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Swiss Legal Assistant (Phi-2.5)")
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(height=500)
            query_box = gr.Textbox(label="Ask a legal question", placeholder="What are the data protection requirements?")
        with gr.Column():
            gr.Markdown("### Retrieved Documents")
            doc_display = gr.Textbox(label="Context Used", interactive=False, lines=10)
    
    def respond(query, history):
        docs = retriever.retrieve(query)
        context = "\n\n".join([f"{i+1}. Document {d['id']} - {d['title']}" for i,d in enumerate(docs)])
        response = generate_response(query)
        history.append((query, response))
        return history, context
    
    query_box.submit(
        respond,
        [query_box, chatbot],
        [chatbot, doc_display]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)
