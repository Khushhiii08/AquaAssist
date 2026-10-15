import os
import glob
import re
import fitz  # PyMuPDF

def sanitize_pdf_text(raw_text):
    """
    Cleans PDF formatting artifacts so the NLP parser can accurately detect grammatical clauses.
    """
    # 1. Fix hyphenated words broken across lines (e.g., "envi-\nronment" -> "environment")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", raw_text)
    
    # 2. Remove mid-sentence line breaks, replacing them with a space
    # (Matches a newline that does NOT follow a period, question mark, or exclamation point)
    text = re.sub(r'(?<![.?!])\n+', ' ', text)
    
    # 3. Collapse multiple spaces into a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 4. Clean up stray empty lines while preserving actual paragraph breaks
    text = re.sub(r'\n{2,}', '\n\n', text)
    
    return text.strip()

def load_and_sanitize_pdfs(pdf_directory="data/raw_pdfs"):
    """
    Scans the directory, extracts raw text page-by-page, and applies NLP sanitization.
    """
    print(f"[*] Scanning '{pdf_directory}' for ICAR-CIBA PDFs...")
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    
    if not pdf_files:
        print(f"[!] No PDFs found. Please check '{pdf_directory}'.")
        return ""

    corpus_text = ""
    for pdf_path in pdf_files:
        try:
            print(f"[*] Extracting and sanitizing: {os.path.basename(pdf_path)}")
            doc = fitz.open(pdf_path)
            
            doc_text = ""
            for page in doc:
                doc_text += page.get_text("text") + "\n"
                
            corpus_text += sanitize_pdf_text(doc_text) + "\n\n"
            
        except Exception as e:
            print(f"[!] Failed to parse {os.path.basename(pdf_path)}: {e}")
            
    print(f"[*] Extraction complete. Corpus length: {len(corpus_text)} characters.")
    return corpus_text

# Quick test block to verify Task 1.1 executes cleanly
if __name__ == "__main__":
    test_text = load_and_sanitize_pdfs("data/raw_pdfs")
    if test_text:
        print("\n[Sample Snippet]:")
        print(test_text[:500])