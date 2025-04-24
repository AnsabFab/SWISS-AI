import os
import faiss
import torch
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
        self.user_info = {}
        self.show_references = False

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
        
        # Create context-aware prompt with retrieved documents
        prompt = self._create_prompt(query, relevant_docs)
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=800,  # Increased token count for more comprehensive responses
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2  # Helps avoid repetitive text
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

ASSISTANT:"""
        
        return prompt

    def _format_references(self, docs: List[Dict]) -> str:
        """Format document references for display"""
        if not docs:
            return "No specific legal references found."
        
        references = []
        for i, doc in enumerate(docs[:5], 1):  # Show up to 5 references
            references.append(f"{i}. {doc['id']}: {doc['title']} ({doc['jurisdiction']} jurisdiction)")
        
        return "\n".join(references)

    def start_conversation(self):
        print("\n=== Swiss Federal Legal Assistant ===")
        print("Type 'references on' to enable document references")
        print("Type 'references off' to disable document references")
        print("Type 'show references' to see relevant documents for your last query")
        print("Type 'quit' to exit\n")
        
        # Collect user information
        self.user_info = {
            "name": input("Your name: ").strip(),
            "age": input("Your age: ").strip(),
            "jurisdiction": self._get_valid_jurisdiction()
        }
        
        print(f"\nWelcome {self.user_info['name']}! How can I assist you with {self.user_info['jurisdiction']} legal matters today?")
        
        last_documents = []
        
        while True:
            try:
                query = input("\nYou: ").strip()
                
                if query.lower() == 'quit':
                    print("Goodbye! Have a nice day.")
                    break
                    
                elif query.lower() == 'references on':
                    self.show_references = True
                    print("Document references enabled. I'll include legal citations in my responses.")
                    continue
                    
                elif query.lower() == 'references off':
                    self.show_references = False
                    print("Document references disabled. I'll provide general legal guidance without citations.")
                    continue
                    
                elif query.lower() == 'show references':
                    if last_documents:
                        print("\nRelevant legal references for your last query:")
                        print(self._format_references(last_documents))
                    else:
                        print("No previous query to show references for.")
                    continue
                
                # Generate response and get relevant documents
                response, last_documents = self._generate_response(query)
                
                print(f"\nAssistant: {response}")
                
                # Show references automatically if enabled
                if self.show_references and last_documents:
                    print("\nReferences:")
                    print(self._format_references(last_documents))
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                import traceback
                print(traceback.format_exc())

    def _get_valid_jurisdiction(self) -> str:
        while True:
            jurisdiction = input("Jurisdiction of interest (Federal/Cantonal/Municipal): ").strip().capitalize()
            if jurisdiction in ["Federal", "Cantonal", "Municipal"]:
                return jurisdiction
            print("Please enter a valid jurisdiction (Federal, Cantonal, or Municipal)")

if __name__ == "__main__":
    assistant = LegalAssistant()
    assistant.start_conversation()
