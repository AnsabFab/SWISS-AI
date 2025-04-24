import os
import faiss
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

class LegalAssistant:
    def __init__(self):
        # Configuration
        self.MODEL_PATH = "/data/models_weights/phi-2-legal-finetuned/"
        self.DOCUMENTS_FILE = "fedlex_documents.txt"
        self.EMBEDDER_NAME = "mixedbread-ai/mxbai-embed-large-v1"
        self.FAISS_INDEX_PATH = "/data/scripts/legal_docs_faiss_index.index"
        
        # Initialize components
        self._initialize_models()
        self._load_documents()
        self._initialize_retriever()
        
        # Conversation context
        self.conversation_history = []
        self.user_info = {"name": "Anonymous", "age": "Not specified", "jurisdiction": "Federal"}
        self.show_references = False
        self.last_relevant_docs = []

    def _initialize_models(self):
        print("Loading language model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.embedder = SentenceTransformer(self.EMBEDDER_NAME)

    def _load_documents(self):
        print("Loading legal documents...")
        self.documents = []
        try:
            with open(self.DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for doc in content.split("---\n"):
                if doc.strip():
                    parts = doc.split("\n", 2)
                    if len(parts) >= 3:
                        self.documents.append({
                            "id": parts[0].strip(),
                            "title": parts[1].strip(),
                            "content": parts[2].strip(),
                            "jurisdiction": self._detect_jurisdiction(parts[1])
                        })
            print(f"Loaded {len(self.documents)} documents")
        except Exception as e:
            print(f"Error loading documents: {str(e)}")

    def _detect_jurisdiction(self, title: str) -> str:
        title_lower = title.lower()
        if any(k in title_lower for k in ["kanton", "kantonsverfassung", "stand"]):
            return "Cantonal"
        elif any(k in title_lower for k in ["gemeinde", "stadt", "stadtverfassung"]):
            return "Municipal"
        return "Federal"

    def _initialize_retriever(self):
        print("Initializing document retriever...")
        if os.path.exists(self.FAISS_INDEX_PATH):
            self.index = faiss.read_index(self.FAISS_INDEX_PATH)
            # Verify embedding dimensions match
            if self.index.d != self.embedder.get_sentence_embedding_dimension():
                print("Rebuilding index due to dimension mismatch...")
                self._build_index()
        else:
            self._build_index()

    def _build_index(self):
        embeddings = self.embedder.encode(
            [doc["content"] for doc in self.documents],
            show_progress_bar=True
        )
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        faiss.write_index(self.index, self.FAISS_INDEX_PATH)

    def _retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve the most relevant documents for a query"""
        query_embed = self.embedder.encode([query])
        distances, indices = self.index.search(query_embed, top_k)
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]

    def _extract_model_response(self, full_response: str) -> str:
        """Extract just the assistant's response from the full generated text"""
        # Split on "ASSISTANT:" and take the last part
        parts = full_response.split("ASSISTANT:")
        if len(parts) > 1:
            return parts[-1].strip()
        return full_response.strip()

    def _generate_response(self, query: str) -> Tuple[str, List[Dict]]:
        """Generate a response to the user query with optional references"""
        # Retrieve relevant documents
        relevant_docs = self._retrieve_documents(query)
        self.last_relevant_docs = relevant_docs
        
        # Create context-aware prompt with retrieved documents
        prompt = self._create_prompt(query, relevant_docs)
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = self._extract_model_response(full_response)
        
        # Update conversation history
        self.conversation_history.append(f"User: {query}")
        self.conversation_history.append(f"Assistant: {response}")
        
        # Return both the response and relevant documents
        return response, relevant_docs

    def _create_prompt(self, query: str, relevant_docs: List[Dict]) -> str:
        """Create a context-rich prompt with conversation history and relevant legal documents"""
        # Format conversation history
        history_text = "\n".join(self.conversation_history[-10:]) if self.conversation_history else ""
        
        # Extract content from relevant documents for context
        doc_excerpts = []
        for doc in relevant_docs[:3]:  # Limit to top 3 for context
            # Take first 500 chars of each document as excerpt
            excerpt = doc["content"][:500] + "..." if len(doc["content"]) > 500 else doc["content"]
            doc_excerpts.append(f"Document ID: {doc['id']}\nTitle: {doc['title']}\nExcerpt: {excerpt}")
        
        legal_context = "\n\n".join(doc_excerpts)
        
        prompt = f"""You are a Swiss legal assistant providing professional legal guidance.
Follow these instructions carefully:
1. Use formal but accessible language
2. Be precise with legal terminology
3. Answer based on Swiss law and the provided legal documents
4. {"Include specific document references in your answer" if self.show_references else "Do not explicitly mention document references unless directly asked"}

USER PROFILE:
Name: {self.user_info.get('name', 'Anonymous')}
Age: {self.user_info.get('age', 'Not specified')}
Jurisdiction of interest: {self.user_info.get('jurisdiction', 'Federal')}

RELEVANT LEGAL DOCUMENTS:
{legal_context}

CONVERSATION HISTORY:
{history_text}

CURRENT QUERY: {query}

A:"""
        
        return prompt

    def _format_references(self, docs: List[Dict]) -> str:
        """Format document references for display"""
        if not docs:
            return "No specific legal references found."
        
        references = []
        for i, doc in enumerate(docs[:5], 1):  # Show up to 5 references
            references.append(f"{i}. {doc['id']}: {doc['title']} ({doc['jurisdiction']} jurisdiction)")
        
        return "\n".join(references)

    def set_user_info(self, name, age, jurisdiction):
        """Set user information"""
        self.user_info = {
            "name": name if name.strip() else "Anonymous",
            "age": age if age.strip() else "Not specified",
            "jurisdiction": jurisdiction
        }
        return f"Welcome {self.user_info['name']}! How can I assist you with {self.user_info['jurisdiction']} legal matters today?"

    def toggle_references(self, show_refs):
        """Toggle showing references"""
        self.show_references = show_refs
        if show_refs:
            return "Document references enabled. I'll include legal citations in my responses."
        else:
            return "Document references disabled. I'll provide general legal guidance without citations."
    
    def chat(self, message, history):
        """Process user message and return response for Gradio chat interface"""
        response, relevant_docs = self._generate_response(message)
        
        # If references are enabled, append them to the response
        if self.show_references and relevant_docs:
            references = self._format_references(relevant_docs)
            response += f"\n\n**References:**\n{references}"
            
        return response
    
    def show_last_references(self):
        """Show references for the last query"""
        if not self.last_relevant_docs:
            return "No previous query to show references for."
        
        return self._format_references(self.last_relevant_docs)

# Initialize the Legal Assistant
assistant = LegalAssistant()

# Define Gradio interface
with gr.Blocks(title="Swiss Legal Assistant", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Swiss Federal Legal Assistant")
    gr.Markdown("Get professional legal guidance based on Swiss law")
    
    with gr.Row():
        with gr.Column(scale=1):
            # User info section
            with gr.Group():
                gr.Markdown("### User Information")
                name_input = gr.Textbox(label="Your Name", placeholder="Enter your name")
                age_input = gr.Textbox(label="Your Age", placeholder="Enter your age")
                jurisdiction_input = gr.Dropdown(
                    ["Federal", "Cantonal", "Municipal"], 
                    label="Jurisdiction of Interest",
                    value="Federal"
                )
                submit_info_btn = gr.Button("Set User Info", variant="primary")
                info_output = gr.Textbox(label="Status", interactive=False)
            
            # Settings section
            with gr.Group():
                gr.Markdown("### Settings")
                references_toggle = gr.Checkbox(label="Show References in Responses", value=False)
                ref_status = gr.Textbox(label="Reference Status", interactive=False)
                show_refs_btn = gr.Button("Show References for Last Query")
                refs_output = gr.Textbox(label="References", interactive=False)
        
        with gr.Column(scale=2):
            # Chat interface
            chatbot = gr.Chatbot(height=600, show_label=False, type="messages")
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask your legal question here...", 
                    container=False,
                    scale=7,
                    show_label=False
                )
                submit_btn = gr.Button("Submit", variant="primary", scale=1)
            
            clear_btn = gr.ClearButton([chatbot, msg_input], value="Clear Chat")

    # Set up events
    submit_info_btn.click(
        assistant.set_user_info, 
        [name_input, age_input, jurisdiction_input], 
        [info_output]
    )
    
    references_toggle.change(
        assistant.toggle_references,
        [references_toggle],
        [ref_status]
    )
    
    show_refs_btn.click(
        assistant.show_last_references,
        [],
        [refs_output]
    )
    
    # Chat functionality
    def user_submit(message, history):
        if message.strip() == "":
            return "", history
        history.append({"role": "user", "content": message})
        return "", history
    
    def bot_response(history):
        if not history:
            return history
        
        user_message = history[-1]["content"]
        response = assistant.chat(user_message, history)
        history.append({"role": "assistant", "content": response})
        return history
    
    msg_input.submit(
        user_submit,
        [msg_input, chatbot],
        [msg_input, chatbot],
        queue=False
    ).then(
        bot_response,
        [chatbot],
        [chatbot]
    )
    
    submit_btn.click(
        user_submit,
        [msg_input, chatbot],
        [msg_input, chatbot],
        queue=False
    ).then(
        bot_response,
        [chatbot],
        [chatbot]
    )
    
    # Print startup message directly instead of using on_load
    print("Swiss Legal Assistant Gradio interface is running!")

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)