import os
import faiss
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional, Any

# --- Configuration ---
# Consider moving these to environment variables or a config file for larger projects
MODEL_PATH = "/data/models_weights/phi-2-legal-finetuned/" # Ensure this path is correct
DOCUMENTS_FILE = "fedlex_documents.txt"                   # Ensure this file exists and is readable
EMBEDDER_NAME = "mixedbread-ai/mxbai-embed-large-v1"
FAISS_INDEX_PATH = "/data/scripts/legal_docs_faiss_index.index" # Ensure this path is writable/readable
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Legal Assistant Class ---
class LegalAssistant:
    """
    A RAG-based legal assistant specialized in Swiss law using a fine-tuned model.
    """
    def __init__(self):
        print("Initializing Legal Assistant...")
        # Configuration passed during initialization
        self.model_path = MODEL_PATH
        self.documents_file = DOCUMENTS_FILE
        self.embedder_name = EMBEDDER_NAME
        self.faiss_index_path = FAISS_INDEX_PATH

        # Check if documents file exists before proceeding
        if not os.path.exists(self.documents_file):
            raise FileNotFoundError(
                f"Critical Error: Documents file not found at '{self.documents_file}'. "
                "The application cannot start without the legal documents."
            )

        # Initialize components
        self._initialize_models()
        self._load_documents()
        self._initialize_retriever()

        # State variables
        self.user_info = {"name": "Anonymous", "age": "Not specified", "jurisdiction": "Federal"}
        self.show_references = False
        self.last_relevant_docs: List[Dict] = []
        # Note: Gradio's ChatInterface manages its own history.
        # This internal history can be used for more complex context management if needed,
        # but for this prompt structure, we rely on the history passed by Gradio.
        # self.conversation_history = []

        print("Legal Assistant Initialized Successfully.")

    def _initialize_models(self):
        """Loads the language model, tokenizer, and sentence embedder."""
        print(f"Loading language model from: {self.model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # Set pad token if it's not already set (common with some models like Phi)
            if self.tokenizer.pad_token is None:
                 self.tokenizer.pad_token = self.tokenizer.eos_token
                 print("Set tokenizer pad_token to eos_token.")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto", # Automatically use available GPUs/CPU
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32, # Use float16 only on GPU
                trust_remote_code=True # Needed for some models like Phi
            )
            print(f"Language model loaded on device: {self.model.device}")

            print(f"Loading sentence embedder: {self.embedder_name}")
            self.embedder = SentenceTransformer(self.embedder_name, device=DEVICE)
            print(f"Sentence embedder loaded on device: {self.embedder.device}")

        except Exception as e:
            print(f"Error loading models: {e}")
            raise  # Re-raise critical error

    def _load_documents(self):
        """Loads and parses legal documents from the specified file."""
        print(f"Loading legal documents from: {self.documents_file}")
        self.documents: List[Dict[str, Any]] = []
        try:
            with open(self.documents_file, 'r', encoding='utf-8') as f:
                content = f.read()

            raw_docs = content.split("---\n")
            print(f"Found {len(raw_docs)} potential document sections.")
            parsed_count = 0
            for idx, doc_text in enumerate(raw_docs):
                if doc_text and doc_text.strip():
                    parts = doc_text.strip().split("\n", 2)
                    if len(parts) >= 3:
                        doc_id = parts[0].strip()
                        title = parts[1].strip()
                        doc_content = parts[2].strip()
                        jurisdiction = self._detect_jurisdiction(title)

                        self.documents.append({
                            "id": doc_id,
                            "title": title,
                            "content": doc_content,
                            "jurisdiction": jurisdiction
                            # Add original index for stable referencing if needed later
                            # "original_index": idx
                        })
                        parsed_count += 1
                    # else:
                    #     print(f"Skipping malformed document section (index {idx}): '{doc_text[:100]}...'")

            print(f"Successfully loaded and parsed {len(self.documents)} documents.")
            if not self.documents:
                 print("Warning: No documents were loaded. Retrieval will not function.")

        except FileNotFoundError:
             # This case is handled by the check in __init__, but kept for safety
             print(f"Error: Documents file not found at {self.documents_file}")
             raise
        except Exception as e:
            print(f"Error loading or parsing documents: {e}")
            # Decide if this is critical - perhaps allow running without docs?
            # For a legal assistant, documents are likely essential.
            raise # Re-raise critical error

    def _detect_jurisdiction(self, title: str) -> str:
        """Attempts to determine jurisdiction based on keywords in the title."""
        title_lower = title.lower()
        # More specific keywords might be needed depending on the data
        if any(k in title_lower for k in ["kanton", "kantonsverfassung", "standes", "interkantonal"]):
            return "Cantonal"
        elif any(k in title_lower for k in ["gemeinde", "stadt", "stadtverfassung", "kommunal"]):
            return "Municipal"
        # Default to Federal if no specific keywords found
        return "Federal"

    def _initialize_retriever(self):
        """Initializes the FAISS index for document retrieval, building it if necessary."""
        print("Initializing document retriever...")
        if not self.documents:
            print("Skipping retriever initialization as no documents were loaded.")
            self.index = None
            return

        embedding_dim = self.embedder.get_sentence_embedding_dimension()
        if embedding_dim is None:
             raise ValueError("Could not determine embedding dimension from the embedder.")

        if os.path.exists(self.faiss_index_path):
            try:
                print(f"Loading existing FAISS index from: {self.faiss_index_path}")
                self.index = faiss.read_index(self.faiss_index_path)
                if self.index.d != embedding_dim:
                    print(f"Index dimension ({self.index.d}) does not match embedder dimension ({embedding_dim}). Rebuilding index.")
                    self._build_index(embedding_dim)
                elif self.index.ntotal != len(self.documents):
                     print(f"Index size ({self.index.ntotal}) does not match number of documents ({len(self.documents)}). Rebuilding index.")
                     self._build_index(embedding_dim)
                else:
                     print("FAISS index loaded successfully.")
            except Exception as e:
                print(f"Error loading FAISS index: {e}. Attempting to rebuild.")
                self._build_index(embedding_dim)
        else:
            print("No existing FAISS index found. Building new index.")
            self._build_index(embedding_dim)

    def _build_index(self, embedding_dim: int):
        """Builds the FAISS index from document contents."""
        print(f"Building FAISS index for {len(self.documents)} documents (Dimension: {embedding_dim})...")
        try:
            # Encode documents in batches for potentially lower memory usage
            # Adjust batch_size based on available memory
            embeddings = self.embedder.encode(
                [doc["content"] for doc in self.documents],
                show_progress_bar=True,
                batch_size=32 # Adjust as needed
            )
            print(f"Embeddings generated with shape: {embeddings.shape}")

            # Create FAISS index - IndexFlatL2 is simple; consider IndexIVFFlat for very large datasets
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.index.add(embeddings) # Add embeddings to the index

            print(f"FAISS index built with {self.index.ntotal} vectors.")

            # Save the index
            print(f"Saving FAISS index to: {self.faiss_index_path}")
            faiss.write_index(self.index, self.faiss_index_path)
            print("FAISS index saved successfully.")

        except Exception as e:
            print(f"Error building or saving FAISS index: {e}")
            self.index = None # Ensure index is None if building failed
            # Decide if this is critical - RAG won't work without an index.
            # raise # Optional: re-raise if index is essential

    def _retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieves the most relevant documents for a query using FAISS."""
        if self.index is None or self.index.ntotal == 0:
            print("Retriever Warning: No index available or index is empty. Skipping retrieval.")
            return []
        if not query or not query.strip():
             print("Retriever Warning: Empty query provided. Skipping retrieval.")
             return []

        try:
            print(f"Retrieving documents for query: '{query[:100]}...'")
            query_embed = self.embedder.encode([query], normalize_embeddings=True) # Normalize for L2/cosine similarity
            distances, indices = self.index.search(query_embed, top_k)

            # Filter out potential invalid indices (though unlikely with IndexFlatL2 if built correctly)
            valid_indices = [i for i in indices[0] if 0 <= i < len(self.documents)]
            retrieved_docs = [self.documents[i] for i in valid_indices]

            print(f"Retrieved {len(retrieved_docs)} documents.")
            self.last_relevant_docs = retrieved_docs # Store for potential later use
            return retrieved_docs
        except Exception as e:
            print(f"Error during document retrieval: {e}")
            return []

    def _extract_model_response(self, full_response: str, prompt: str) -> str:
        """
        Extracts only the newly generated part of the response, excluding the prompt.
        Also tries to remove common instruction-following artifacts.
        """
        # Basic extraction: remove the prompt
        response = full_response[len(prompt):].strip()

        # More robust extraction for models that might repeat parts of the prompt structure
        # Find the marker we used to signal the start of the assistant's response
        assistant_marker = "ASSISTANT:"
        marker_pos = response.find(assistant_marker)
        if marker_pos != -1:
            response = response[marker_pos + len(assistant_marker):].strip()
        
        # Sometimes models start with "A:" or similar, remove that too if it's left
        if response.lower().startswith("a:"):
             response = response[2:].strip()

        # Remove potential initial pleasantries if they are repetitive
        # common_greetings = ["Okay, I can help with that.", "Certainly, here's the information:", "Based on the documents:"]
        # for greeting in common_greetings:
        #      if response.startswith(greeting):
        #           response = response[len(greeting):].strip()
        #           break # Only remove one

        return response

    def _create_prompt(self, query: str, history: List[List[str]], relevant_docs: List[Dict]) -> str:
        """
        Creates a detailed, structured prompt incorporating persona, instructions,
        context (user info, history, docs), and the current query.
        Uses prompt engineering techniques for domain restriction and clarity.
        """
        # --- Persona and Core Directive ---
        prompt_parts = [
            "### Role Definition:",
            "You are a specialized Swiss Legal Information Assistant. Your *sole* purpose is to provide information and answer questions strictly related to Swiss law, based ONLY on the provided legal document excerpts and context. Maintain a professional, objective, and neutral tone. You are an AI assistant, not a human lawyer."
        ]

        # --- Strict Instructions & Constraints ---
        prompt_parts.extend([
            "\n### Core Instructions:",
            "1.  **Domain Restriction:** Answer ONLY questions about Swiss law (federal, cantonal, municipal). If a question is outside this scope (e.g., foreign law, medical advice, financial advice, personal opinions, general knowledge), you MUST explicitly state that you cannot answer because it's outside your designated function as a Swiss legal assistant. Do NOT attempt to answer off-topic questions.",
            "2.  **Information Source:** Base your answers PRIMARILY on the 'RELEVANT LEGAL DOCUMENTS' provided below. If the documents don't contain relevant information, state that the provided texts do not cover the query.",
            "3.  **Clarity and Structure:** Provide clear, concise, and well-structured answers. Use bullet points or numbered lists where appropriate for readability.",
            "4.  **Precision:** Be precise with legal concepts. Avoid speculation or providing definitive legal advice.",
            f"5.  **References:** {'If relevant documents were found, briefly mention the source document ID(s) (e.g., "According to document SR 123...") within your answer.' if self.show_references else 'Do not explicitly cite document IDs unless directly asked, but use the information.'}",
            "6.  **Disclaimer:** ALWAYS conclude your response with the mandatory disclaimer: 'Disclaimer: This is AI-generated information based on the provided texts and does not constitute legal advice. Consult with a qualified legal professional for advice specific to your situation.'",
            "7.  **Tone:** Formal, objective, and helpful within the defined scope.",
            "8.  **Language:** Respond in the language of the query if possible (primarily English, German, French, Italian supported by the underlying model, but focus on matching query language).",
            "9.  **Safety:** Do not generate harmful, unethical, or offensive content. Refuse inappropriate requests politely."
        ])

        # --- Contextual Information ---
        prompt_parts.extend([
            "\n### User Profile Context:",
            f"- Name: {self.user_info.get('name', 'Anonymous')}",
            f"- Age: {self.user_info.get('age', 'Not specified')}",
            f"- Jurisdiction Focus: {self.user_info.get('jurisdiction', 'Federal')}"
        ])

        prompt_parts.append("\n### Relevant Legal Documents Context:")
        if relevant_docs:
            doc_excerpts = []
            # Limit context length to avoid exceeding model limits
            # Use first ~500 chars, prioritize docs matching jurisdiction focus? (More complex logic)
            for doc in relevant_docs[:3]: # Limit to top 3-5 docs for context window
                excerpt = doc["content"][:700] + ("..." if len(doc["content"]) > 700 else "")
                doc_excerpts.append(
                    f"--- Document Start ---\n"
                    f"ID: {doc.get('id', 'N/A')}\n"
                    f"Title: {doc.get('title', 'N/A')}\n"
                    f"Jurisdiction: {doc.get('jurisdiction', 'N/A')}\n"
                    f"Excerpt: {excerpt}\n"
                    f"--- Document End ---"
                )
            prompt_parts.append("\n\n".join(doc_excerpts))
        else:
            prompt_parts.append("No specific relevant documents were retrieved for this query.")

        prompt_parts.append("\n### Conversation History (Last 5 Turns):")
        if history:
            # Format history clearly for the model
            history_text = []
            # Take last 5 user/assistant pairs
            for user_msg, assistant_msg in history[-5:]:
                 history_text.append(f"USER: {user_msg}")
                 history_text.append(f"ASSISTANT: {assistant_msg}")
            prompt_parts.append("\n".join(history_text))
        else:
            prompt_parts.append("No previous conversation history.")

        # --- The Actual Query ---
        prompt_parts.extend([
            "\n### Current User Query:",
            query
        ])

        # --- Response Trigger ---
        # Use a clear marker to signal where the model should start its response
        prompt_parts.append("\n### Assistant Response:\nASSISTANT:")

        return "\n".join(prompt_parts)


    def _generate_response(self, query: str, history: List[List[str]]) -> Tuple[str, List[Dict]]:
        """
        Generates a response using RAG: retrieves documents, creates prompt,
        calls the LLM, and extracts the answer.
        """
        # 1. Retrieve relevant documents
        relevant_docs = self._retrieve_documents(query, top_k=5)

        # 2. Create the detailed prompt
        prompt = self._create_prompt(query, history, relevant_docs)
        # print(f"\n--- PROMPT --- \n{prompt}\n--- END PROMPT ---\n") # DEBUG: Print the prompt

        # 3. Generate response using the LLM
        response_text = "Error: Could not generate response." # Default error message
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device) # Adjust max_length based on model
            
            # Generation parameters - tune these for desired output style
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=600,       # Max length of the generated response part
                temperature=0.6,          # Lower for more factual, higher for creativity
                top_p=0.9,                # Nucleus sampling
                do_sample=True,           # Enable sampling
                repetition_penalty=1.15,  # Penalize repeating tokens
                pad_token_id=self.tokenizer.eos_token_id # Important for open-ended generation
            )
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 4. Extract only the assistant's generated part
            response_text = self._extract_model_response(full_response, prompt)

            # 5. Post-processing (ensure disclaimer is present, could add more checks)
            disclaimer = "Disclaimer: This is AI-generated information based on the provided texts and does not constitute legal advice. Consult with a qualified legal professional for advice specific to your situation."
            if disclaimer not in response_text:
                 print("Warning: Disclaimer missing from initial generation. Appending manually.")
                 response_text += f"\n\n{disclaimer}"

        except Exception as e:
            print(f"Error during model generation: {e}")
            response_text = f"Sorry, I encountered an error while generating the response. Please try again. Error details: {str(e)}"
            # Ensure disclaimer is added even to error messages
            response_text += f"\n\nDisclaimer: This is AI-generated information..."


        # print(f"\n--- RESPONSE --- \n{response_text}\n--- END RESPONSE ---\n") # DEBUG: Print the response

        # Return the response and the documents used (primarily for reference display)
        return response_text, relevant_docs

    def _format_references(self, docs: List[Dict]) -> str:
        """Formats retrieved document information for display."""
        if not docs:
            return "No specific legal references were retrieved for the last query."

        references = ["**Retrieved References:**"]
        for i, doc in enumerate(docs[:5], 1): # Show top 5 references
            references.append(
                f"{i}. **ID:** {doc.get('id', 'N/A')}\n"
                f"   **Title:** {doc.get('title', 'N/A')}\n"
                f"   **Jurisdiction:** {doc.get('jurisdiction', 'N/A')}"
                # Add a snippet? f"   **Excerpt:** {doc.get('content', '')[:100]}..."
            )
        return "\n".join(references)

    # --- Public Methods for Gradio Interface ---

    def set_user_info(self, name: str, age: str, jurisdiction: str) -> str:
        """Updates user information state."""
        self.user_info = {
            "name": name.strip() if name and name.strip() else "Anonymous",
            "age": age.strip() if age and age.strip() else "Not specified",
            "jurisdiction": jurisdiction
        }
        print(f"User info updated: {self.user_info}")
        return f"User profile updated. Welcome {self.user_info['name']}! Focus set to {self.user_info['jurisdiction']} jurisdiction."

    def toggle_references(self, show_refs: bool) -> str:
        """Toggles the display of references in responses."""
        self.show_references = show_refs
        status = "ENABLED" if show_refs else "DISABLED"
        print(f"Reference display set to: {status}")
        if show_refs:
            return "Reference display ENABLED. Responses will attempt to cite source document IDs."
        else:
            return "Reference display DISABLED. Responses will focus on information without explicit citations."

    def chat_interface_logic(self, message: str, history: List[List[str]]) -> str:
        """
        Main logic handler for the Gradio ChatInterface.
        Receives user message and history, returns the assistant's response.
        """
        if not message or not message.strip():
             return "Please enter a question."

        print(f"Processing query: '{message[:100]}...'")
        # Generate the response using the core RAG logic
        response, _ = self._generate_response(message, history) # We don't need docs directly here

        # The references toggle now controls inclusion *within* the prompt/response generation
        # We no longer need to append them separately here, simplifying this method.

        # Gradio ChatInterface automatically updates history based on the return value
        return response

    def show_last_references_info(self) -> str:
        """Returns formatted references from the *last* query processed."""
        print("Showing references for the last processed query.")
        if not self.last_relevant_docs:
            return "No references found for the last query, or no query has been processed yet."
        return self._format_references(self.last_relevant_docs)

# --- Gradio App Definition ---

def build_gradio_app(assistant: LegalAssistant) -> gr.Blocks:
    """Builds the Gradio interface."""

    with gr.Blocks(title="Swiss Legal Assistant", theme=gr.themes.Soft(primary_hue="blue", secondary_hue="neutral")) as app:
        gr.Markdown(
            """
            #???? Swiss Legal Information Assistant (AI Prototype)
            Ask questions about Swiss federal, cantonal, or municipal law.
            **Disclaimer:** This AI provides information based on loaded legal texts and does not constitute legal advice. Always consult a qualified legal professional for specific situations. Responses are generated by an AI and may contain inaccuracies.
            """
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                 with gr.Accordion("Settings & User Profile", open=False):
                    with gr.Group():
                        gr.Markdown("### User Information")
                        name_input = gr.Textbox(label="Your Name (Optional)", placeholder="Enter your name", value=assistant.user_info["name"])
                        age_input = gr.Textbox(label="Your Age (Optional)", placeholder="Enter your age", value=assistant.user_info["age"])
                        jurisdiction_input = gr.Dropdown(
                            choices=["Federal", "Cantonal", "Municipal"],
                            label="Jurisdiction Focus",
                            value=assistant.user_info["jurisdiction"]
                        )
                        submit_info_btn = gr.Button("Update User Info", variant="secondary")
                        info_output = gr.Textbox(label="User Info Status", interactive=False, lines=2)

                    with gr.Group():
                        gr.Markdown("### Assistant Settings")
                        references_toggle = gr.Checkbox(
                            label="Cite Document IDs in Responses",
                            value=assistant.show_references,
                            info="Check to include references like (SR 123) in the text."
                        )
                        ref_status_output = gr.Textbox(label="Reference Setting Status", interactive=False, lines=2)
                        show_refs_btn = gr.Button("Show References for Last Query", variant="secondary")
                        last_refs_output = gr.Textbox(label="Last Query References", interactive=False, lines=5)

            with gr.Column(scale=3):
                 # Use ChatInterface for a streamlined chat UI
                 # REMOVED arguments: submit_btn, clear_btn, retry_btn, undo_btn
                 # ADDED type='messages' to gr.Chatbot component definition
                 chat_interface = gr.ChatInterface( # Renamed variable for clarity
                      fn=assistant.chat_interface_logic, # Core function processing messages
                      chatbot=gr.Chatbot(
                           label="Legal Assistant Chat",
                           height=650,
                           show_label=False,
                           # Explicitly set type='messages' as recommended by the warning
                           type='messages',
                           avatar_images=(None, "https://img.icons8.com/fluency/96/scales.png") # Optional: User/Bot avatars
                           ),
                      textbox=gr.Textbox(placeholder="Ask your Swiss legal question here...", container=False, scale=7),
                      # Rely on default buttons provided by ChatInterface
                      examples=[
                           ["What are the basic requirements for Swiss citizenship?", "Federal"],
                           ["Explain the concept of 'dual criminality' in Swiss extradition law.", "Federal"],
                           ["What is the difference between 'Obligationenrecht' and 'Zivilgesetzbuch'?", "Federal"],
                           ["Can a cantonal law contradict a federal law in Switzerland?", "Cantonal"],
                           # Add more specific examples relevant to your documents
                      ],
                      # Removed problematic arguments here
                 )

        # --- Event Handlers for Settings ---
        submit_info_btn.click(
            assistant.set_user_info,
            inputs=[name_input, age_input, jurisdiction_input],
            outputs=[info_output]
        )

        references_toggle.change(
            assistant.toggle_references,
            inputs=[references_toggle],
            outputs=[ref_status_output]
        )

        show_refs_btn.click(
            assistant.show_last_references_info,
            inputs=[],
            outputs=[last_refs_output]
        )

        # Initialize status displays on load
        app.load(lambda: assistant.toggle_references(assistant.show_references), outputs=[ref_status_output])
        app.load(lambda: assistant.set_user_info(assistant.user_info["name"], assistant.user_info["age"], assistant.user_info["jurisdiction"]), outputs=[info_output])

    return app

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Initialize the assistant *before* building the app
        legal_assistant = LegalAssistant()

        # Build the Gradio app using the initialized assistant
        gradio_app = build_gradio_app(legal_assistant)

        # Launch the app
        print("Launching Gradio app...")
        gradio_app.queue().launch( # Use queue() for better handling of concurrent users/long generations
            server_name="0.0.0.0", # Listen on all network interfaces
            server_port=7860,
            share=True # Set to True to get a public link (use with caution)
            # debug=True # Enable for Gradio debugging info
        )
    except FileNotFoundError as e:
         print(f"Initialization failed: {e}")
         print("Please ensure the documents file exists and the path is correct.")
    except Exception as e:
         print(f"An unexpected error occurred during startup: {e}")
         # Provide more detailed traceback if needed for debugging
         import traceback
         traceback.print_exc()
