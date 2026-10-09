import os
import glob
import re
import fitz  # PyMuPDF
import spacy
from tqdm import tqdm

# Load English NLP model for semantic parsing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[!] spaCy model not found. Run: python -m spacy download en_core_web_sm")
    exit()

def sanitize_pdf_text(raw_text):
    """Task 1.1: Cleans PDF formatting artifacts."""
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", raw_text)
    text = re.sub(r'(?<![.?!])\n+', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

def load_and_sanitize_pdfs(pdf_directory="data/raw_pdfs"):
    """Task 1.1: Scans directory and extracts sanitized text."""
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    if not pdf_files:
        return ""

    corpus_text = ""
    for pdf_path in pdf_files:
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                corpus_text += sanitize_pdf_text(page.get_text("text")) + "\n\n"
        except Exception as e:
            pass
    return corpus_text

def extract_semantic_triplets(corpus_text):
    """
    Task 1.2: Parses the massive corpus into structured grammatical triplets
    using batch processing to prevent RAM overflow.
    """
    print("[*] Splitting massive corpus into manageable NLP processing blocks...")
    # Split by double newline (paragraphs) to maintain sentence integrity
    paragraphs = corpus_text.split("\n\n")
    
    structured_chunks = []
    
    # Wrap the paragraphs in a tqdm progress bar since NLP parsing takes time
    for para in tqdm(paragraphs, desc="Parsing Semantic Triplets"):
        if len(para.strip()) < 20:  # Skip empty or micro-paragraphs
            continue
            
        doc = nlp(para)
        
        for sent in doc.sents:
            subject = []
            verb = []
            obj = []
            
            # Dependency parsing to isolate core structural components
            for token in sent:
                if "subj" in token.dep_:
                    subject.append(token.text)
                elif "ROOT" in token.dep_ or token.pos_ == "VERB":
                    verb.append(token.text)
                elif "obj" in token.dep_ or "attr" in token.dep_:
                    obj.append(token.text)
            
            # Only create a chunk if a complete logical clause exists
            if subject and verb and obj:
                clean_subject = " ".join(subject)
                clean_verb = " ".join(verb)
                clean_obj = " ".join(obj)
                
                # Format into a direct, concentrated statement
                actionable_chunk = f"Observation: {clean_subject} {clean_verb} {clean_obj}. Context: {sent.text.strip()}"
                structured_chunks.append(actionable_chunk)

    print(f"[*] Successfully extracted {len(structured_chunks)} semantic triplets.")
    return structured_chunks

if __name__ == "__main__":
    print("[*] Starting Task 1.1: Document Parsing & Extraction")
    raw_corpus = load_and_sanitize_pdfs("data/raw_pdfs")
    
    if raw_corpus:
        print("\n[*] Starting Task 1.2: Semantic Triplet Chunking")
        semantic_chunks = extract_semantic_triplets(raw_corpus)
        
        print("\n[Sample Triplet Chunk]:")
        # Print the 50th valid triplet to verify it captured biological context
        if len(semantic_chunks) > 50:
            print(semantic_chunks[50])