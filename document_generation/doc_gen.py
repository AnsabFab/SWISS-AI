import gradio as gr
import os
import tempfile
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors

# Load local model and tokenizer
MODEL_PATH = "/data/models_weights/phi-2-legal-finetuned/"

# Initialize tokenizer and model
print("Loading model and tokenizer from local path...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    device_map="auto",
    load_in_8bit=True  # Use 8-bit quantization for memory efficiency
)

# Create text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=2048,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

# Swiss legal document templates
TEMPLATES = {
    "general_contract": {
        "name": "General Contract",
        "description": "A standard contract template according to Swiss law",
        "sections": [
            "Title", "Parties", "Preamble", "Definitions", "Subject Matter", 
            "Terms and Conditions", "Duration", "Termination", "Governing Law", "Signatures"
        ],
        "prompt": """
Generate a standard contract under Swiss law with the following details:
User Context: {context}
Requirements: {requirements}

The contract should include the following sections:
1. Title of the agreement
2. Parties involved
3. Preamble explaining the purpose
4. Definitions of key terms
5. Subject matter clearly describing what is agreed upon
6. Terms and conditions with clear clauses
7. Duration of the agreement
8. Termination clauses
9. Governing law (Swiss law)
10. Signature lines

Format the contract in a professional legal style.
"""
    },
    "employment_contract": {
        "name": "Employment Contract",
        "description": "Employment agreement according to Swiss labor law",
        "sections": [
            "Title", "Employer Details", "Employee Details", "Position", "Start Date", 
            "Salary", "Working Hours", "Probation Period", "Notice Period", "Confidentiality", 
            "Governing Law", "Signatures"
        ],
        "prompt": """
Generate an employment contract under Swiss labor law with the following details:
User Context: {context}
Requirements: {requirements}

The employment contract should include the following sections:
1. Title
2. Employer details
3. Employee details
4. Position and responsibilities
5. Start date and duration
6. Salary and benefits
7. Working hours
8. Probation period
9. Notice period
10. Confidentiality clause
11. Governing law (Swiss law)
12. Signatures

Ensure compliance with Swiss labor law including references to relevant code articles where appropriate.
"""
    },
    "rental_agreement": {
        "name": "Rental Agreement",
        "description": "Property rental agreement according to Swiss tenancy law",
        "sections": [
            "Title", "Landlord Details", "Tenant Details", "Property Description", 
            "Rental Term", "Rent Amount", "Deposit", "Maintenance", "House Rules", 
            "Termination", "Governing Law", "Signatures"
        ],
        "prompt": """
Generate a rental agreement under Swiss tenancy law with the following details:
User Context: {context}
Requirements: {requirements}

The rental agreement should include the following sections:
1. Title
2. Landlord details
3. Tenant details
4. Property description
5. Rental term (start date, duration)
6. Rent amount and payment terms
7. Deposit information
8. Maintenance responsibilities
9. House rules
10. Termination conditions
11. Governing law (Swiss law)
12. Signatures

Ensure compliance with Swiss tenancy law including references to relevant code articles where appropriate.
"""
    },
    "power_of_attorney": {
        "name": "Power of Attorney",
        "description": "Power of attorney document according to Swiss civil law",
        "sections": [
            "Title", "Principal", "Attorney-in-Fact", "Powers Granted", "Duration", 
            "Revocation", "Governing Law", "Signatures"
        ],
        "prompt": """
Generate a power of attorney document under Swiss civil law with the following details:
User Context: {context}
Requirements: {requirements}

The power of attorney document should include the following sections:
1. Title
2. Principal details
3. Attorney-in-fact details
4. Powers granted
5. Duration/validity period
6. Revocation conditions
7. Governing law (Swiss law)
8. Signatures and notarization requirement

Make sure the document is clear about the scope of powers granted and complies with Swiss civil law.
"""
    },
    "confidentiality_agreement": {
        "name": "Confidentiality Agreement (NDA)",
        "description": "Non-disclosure agreement according to Swiss law",
        "sections": [
            "Title", "Parties", "Purpose", "Definition of Confidential Information", 
            "Obligations", "Exclusions", "Term", "Remedies", "Return of Information", 
            "Governing Law", "Signatures"
        ],
        "prompt": """
Generate a confidentiality agreement (NDA) under Swiss law with the following details:
User Context: {context}
Requirements: {requirements}

The confidentiality agreement should include the following sections:
1. Title
2. Parties involved
3. Purpose of disclosure
4. Definition of confidential information
5. Obligations of receiving party
6. Exclusions from confidential information
7. Term of agreement
8. Remedies for breach
9. Return of confidential information
10. Governing law (Swiss law)
11. Signatures

Ensure the document clearly defines what constitutes confidential information and the obligations to protect it according to Swiss law.
"""
    }
}

def generate_document(template_name, context, requirements):
    """Generate legal document based on selected template and user input"""
    if template_name not in TEMPLATES:
        return "Invalid template selection"
    
    template = TEMPLATES[template_name]
    prompt = template["prompt"].format(context=context, requirements=requirements)
    
    # Generate text using the model
    result = generator(prompt, max_length=2048, do_sample=True, temperature=0.7)[0]["generated_text"]
    
    # Extract the generated document (everything after the prompt)
    document_text = result.split(prompt)[-1].strip()
    
    return document_text

def create_pdf(document_text, template_name):
    """Convert the generated document to PDF"""
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_file.close()
    
    # Set up the document
    doc = SimpleDocTemplate(
        temp_file.name,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Define styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='Justify',
        alignment=TA_JUSTIFY,
        fontSize=10,
        leading=12
    ))
    styles.add(ParagraphStyle(
        name='Title',
        alignment=TA_CENTER,
        fontSize=16,
        leading=18,
        textColor=colors.black,
        spaceAfter=24
    ))
    styles.add(ParagraphStyle(
        name='Heading',
        fontSize=12,
        leading=14,
        textColor=colors.black,
        spaceBefore=12,
        spaceAfter=6
    ))
    
    # Parse the document text into paragraphs
    paragraphs = []
    
    # Add title
    template_info = TEMPLATES[template_name]
    title = template_info["name"].upper()
    paragraphs.append(Paragraph(title, styles["Title"]))
    
    # Add document id and date
    doc_id = f"DOC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    current_date = datetime.now().strftime("%d %B %Y")
    paragraphs.append(Paragraph(f"Document ID: {doc_id}", styles["Normal"]))
    paragraphs.append(Paragraph(f"Generated on: {current_date}", styles["Normal"]))
    paragraphs.append(Spacer(1, 12))
    
    # Add legal disclaimer
    disclaimer = "DISCLAIMER: This document is generated by an AI system for informational purposes only. It should be reviewed by a qualified legal professional before use."
    paragraphs.append(Paragraph(disclaimer, styles["Justify"]))
    paragraphs.append(Spacer(1, 12))
    
    # Process the document text
    sections = document_text.split('\n\n')
    for section in sections:
        if not section.strip():
            continue
            
        # Check if this is a heading (for simple formatting)
        if len(section.strip()) < 100 and not section.endswith('.'):
            paragraphs.append(Paragraph(section, styles["Heading"]))
        else:
            paragraphs.append(Paragraph(section, styles["Justify"]))
            paragraphs.append(Spacer(1, 6))
    
    # Build the PDF
    doc.build(paragraphs)
    
    return temp_file.name

def generate_document_pdf(template_name, context, requirements):
    """Generate document and return as PDF"""
    document_text = generate_document(template_name, context, requirements)
    pdf_path = create_pdf(document_text, template_name)
    
    # Save document history
    save_document_history(template_name, context, requirements, document_text)
    
    return document_text, pdf_path

def update_template_info(template_name):
    """Return information about the selected template"""
    if template_name in TEMPLATES:
        template = TEMPLATES[template_name]
        sections = "\n".join([f"- {section}" for section in template["sections"]])
        return f"### {template['name']}\n\n{template['description']}\n\n**Sections included:**\n{sections}"
    return ""

# Error handling for model loading
def safe_load_model():
    try:
        # Check if model path exists
        if not os.path.exists(MODEL_PATH):
            return f"Error: Model path not found at {MODEL_PATH}"
        return "Model loaded successfully!"
    except Exception as e:
        return f"Error loading model: {str(e)}"

# Function to save generated documents
def save_document_history(template_name, context, requirements, document_text):
    try:
        # Create directory if it doesn't exist
        os.makedirs("document_history", exist_ok=True)
        
        # Create a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"document_history/doc_{template_name}_{timestamp}.json"
        
        # Save document data
        data = {
            "template": template_name,
            "context": context,
            "requirements": requirements,
            "document_text": document_text,
            "timestamp": timestamp
        }
        
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
            
        return True
    except Exception as e:
        print(f"Error saving document history: {str(e)}")
        return False

# Define Gradio interface
with gr.Blocks(title="Swiss Legal Document Generator") as app:
    gr.Markdown("# Swiss Legal Document Generator")
    gr.Markdown("Generate customized legal documents based on Swiss law using a fine-tuned Phi-2 legal model.")
    
    model_status = gr.Markdown(f"Model status: {safe_load_model()}")
    
    with gr.Row():
        with gr.Column(scale=1):
            template_selector = gr.Dropdown(
                choices=list(TEMPLATES.keys()),
                value="general_contract",
                label="Select Document Template"
            )
            template_info = gr.Markdown("Select a template to see details")
            
            context_input = gr.Textbox(
                lines=5,
                placeholder="Enter your context (e.g., your role, situation, relationships, company details, etc.)",
                label="Your Context"
            )
            
            requirements_input = gr.Textbox(
                lines=5,
                placeholder="Enter specific requirements for the document (e.g., terms, dates, special clauses)",
                label="Document Requirements"
            )
            
            generate_button = gr.Button("Generate Document", variant="primary")
        
        with gr.Column(scale=2):
            output_text = gr.Textbox(
                lines=20,
                label="Generated Document",
                placeholder="Generated document will appear here..."
            )
            
            output_pdf = gr.File(label="Download PDF")
    
    template_selector.change(update_template_info, inputs=template_selector, outputs=template_info)
    generate_button.click(
        generate_document_pdf,
        inputs=[template_selector, context_input, requirements_input],
        outputs=[output_text, output_pdf]
    )
    
    share_info = gr.Markdown("""
    ## Share this app
    
    When you launch this application, a public URL will be generated and displayed in the console. 
    You can share this URL with others to give them access to this legal document generator.
    """)
    
    gr.Markdown("""
    ## About this app
    
    This application generates legal documents based on Swiss law using a locally hosted Phi-2 model fine-tuned for legal document generation. Select a template, provide your specific context and requirements, and the application will generate a customized document for you.
    
    ### Example usage:
    
    **Context:** "I am starting a web development business and need to hire a full-time developer."
    
    **Requirements:** "Looking for a 6-month contract with a 3-month probation period, 8,000 CHF monthly salary, 40 hours per week, based in Zurich with intellectual property clauses."
    
    **Note:** While this tool can help create initial drafts, all generated documents should be reviewed by a qualified legal professional before use.
    """)

# Launch the app with public sharing enabled (simplified parameters)
if __name__ == "__main__":
    app.launch(share=True, server_port=7860, server_name="0.0.0.0")
