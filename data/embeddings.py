import os
import random
import pdfplumber
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

# 1. Define PDF directories (same as original)
DATA_DIRS = [
    "/data/law_doc_scraper/entscheidsuche",
    "/data/law_doc_scraper/fedlex",
    "/data/law_doc_scraper/lexfind"
]

# 2. Get a small sample of PDFs (10 per directory)
def get_sample_pdfs(dirs, samples_per_dir=10):
    sample_paths = []
    for dir_path in dirs:
        pdf_paths = []
        for root, _, files in os.walk(dir_path):
            pdfs = [os.path.join(root, f) for f in files if f.endswith(".pdf")]
            pdf_paths.extend(pdfs)
        
        # Randomly sample PDFs (avoid empty folders)
        if pdf_paths:
            sample_paths.extend(random.sample(pdf_paths, min(samples_per_dir, len(pdf_paths))))
    
    return sample_paths

# 3. Text extraction (optimized with error handling)
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return " ".join(page.extract_text() or "" for page in pdf.pages[:3])  # Only first 3 pages for speed
    except Exception as e:
        print(f"Error in {os.path.basename(pdf_path)}: {str(e)[:50]}...")
        return ""

# 4. Process only the sample PDFs
def process_sample():
    print("?? Selecting random sample of PDFs...")
    sample_paths = get_sample_pdfs(DATA_DIRS)
    print(f"Selected {len(sample_paths)} PDFs for testing.")
    
    print("?? Extracting text...")
    texts = []
    for path in tqdm(sample_paths):
        text = extract_text_from_pdf(path)
        if text.strip():
            texts.append(text[:5000])  # Truncate long docs
    
    print("?? Generating embeddings (this may take a few minutes)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Save results
    np.save("legal_embeddings_SAMPLE.npy", embeddings)
    with open("sample_documents.txt", "w") as f:
        f.write("\n\n---\n\n".join([f"DOC {i+1}:\n{t[:300]}..." for i,t in enumerate(texts)]))
    
    print(f"? Saved {len(texts)} embeddings to legal_embeddings_SAMPLE.npy")
    print(f"Sample documents preview saved to sample_documents.txt")

if __name__ == "__main__":
    process_sample()
