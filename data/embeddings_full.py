import os
import pdfplumber
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

# Define PDF directories to analyze
DATA_DIRS = [
    "/data/law_doc_scraper/entscheidsuche",
    "/data/law_doc_scraper/fedlex",
    "/data/law_doc_scraper/lexfind"
]

def get_folder_size(path):
    """Calculate total size of PDFs in a directory (in MB)"""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith('.pdf'):
                total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)  # Convert to MB

def identify_smallest_folder(dirs):
    """Find and confirm the directory with smallest total PDF size"""
    print("\n?? Analyzing folder sizes...")
    sizes = {}
    for dir_path in dirs:
        size = get_folder_size(dir_path)
        sizes[dir_path] = size
        print(f"  - {dir_path}: {size:.2f} MB")
    
    smallest_dir = min(sizes.items(), key=lambda x: x[1])
    print(f"\n? Selected smallest folder: {smallest_dir[0]} ({smallest_dir[1]:.2f} MB)")
    return smallest_dir[0]

def get_all_pdfs_from_folder(folder_path):
    """Get all PDF paths from a single folder"""
    pdf_paths = []
    for root, _, files in os.walk(folder_path):
        pdfs = [os.path.join(root, f) for f in files if f.endswith(".pdf")]
        pdf_paths.extend(pdfs)
    return pdf_paths

def extract_full_text(pdf_path):
    """Extract text from all pages of a PDF"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return " ".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        print(f"?? Error in {os.path.basename(pdf_path)}: {str(e)[:100]}")
        return ""

def process_smallest_folder():
    # 1. Identify target folder
    target_folder = identify_smallest_folder(DATA_DIRS)
    
    # 2. Get all PDFs
    print(f"\n?? Gathering PDFs from {target_folder}...")
    pdf_paths = get_all_pdfs_from_folder(target_folder)
    print(f"  - Found {len(pdf_paths)} documents")
    
    # 3. Process documents
    print("\n?? Extracting text (all pages)...")
    texts = []
    for path in tqdm(pdf_paths, desc="Processing"):
        text = extract_full_text(path)
        if text.strip():
            texts.append(text)
    
    # 4. Generate embeddings
    print("\n?? Creating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # 5. Save results
    output_name = os.path.basename(target_folder)
    np.save(f"{output_name}_embeddings.npy", embeddings)
    with open(f"{output_name}_documents.txt", "w") as f:
        f.write("\n\n---\n\n".join([f"DOC {i+1}:\n{t[:500]}..." for i,t in enumerate(texts)]))
    
    print(f"\n? Successfully processed {len(texts)} documents")
    print(f"  - Embeddings saved to: {output_name}_embeddings.npy")
    print(f"  - Document samples saved to: {output_name}_documents.txt")

if __name__ == "__main__":
    process_smallest_folder()
