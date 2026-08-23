import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "../AI Agent Assessment - Candidate Pack"
DB_DIR = "./chroma_db"

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def build_vectorstore():
    # Initialize Chroma client
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Use default sentence-transformers model
    sentence_transformer_ef = embedding_functions.DefaultEmbeddingFunction()
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="parcelpilot_docs",
        embedding_function=sentence_transformer_ef
    )
    
    # Check if already populated
    if collection.count() > 0:
        print(f"Collection already has {collection.count()} documents. Skipping build.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    documents = []
    metadatas = []
    ids = []
    
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        file_path = os.path.join(DATA_DIR, pdf_file)
        text = extract_text_from_pdf(file_path)
        
        # Determine metadata
        is_deprecated = "DEPRECATED" in pdf_file
        doc_type = "Policy" if "Policy" in pdf_file else "SOP" if "SOP" in pdf_file else "Guide" if "Guide" in pdf_file else "Agreement"
        
        chunks = text_splitter.split_text(text)
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "source": pdf_file,
                "is_deprecated": is_deprecated,
                "doc_type": doc_type,
                "chunk_id": i
            })
            ids.append(f"{pdf_file}_chunk_{i}")
            
    print(f"Adding {len(documents)} chunks to ChromaDB...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("Vectorstore build complete!")

if __name__ == "__main__":
    build_vectorstore()
