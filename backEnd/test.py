import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
# Configuration
GOOGLE_API_KEY = os.getenv("GEMINI_KEY")  # Replace with your actual API key
PDF_PATH = "a.pdf"  # Path to your PDF file
CHROMA_DB_DIR = "chromaDB"  # Directory to store the vector database
COLLECTION_NAME = "pdf_collection"  # Name of the ChromaDB collection
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    print(f"Extracting text from {pdf_path}...")
    loader = PyMuPDF4LLMLoader(pdf_path)
    documents = loader.load()
    print(f"Extracted {len(documents)} document sections")
    return documents

def chunk_text(documents):
    """Split documents into smaller chunks."""
    print("Chunking document text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks")
    return chunks

def create_vector_database(chunks):
    """Create or load a vector database from document chunks."""
    print("Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    print("Checking for existing vector database...")
    # Check if the directory exists and has content indicating a valid DB
    if os.path.exists(CHROMA_DB_DIR) and os.path.exists(os.path.join(CHROMA_DB_DIR, "chroma.sqlite3")):
        # Load existing database
        print("Loading existing vector database...")
        db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME
        )
        print(f"Loaded database with {db.get().get('ids', []).__len__()} documents")
        return db
    else:
        # Create new database
        print("Creating new vector database...")
        # Ensure the directory exists
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_DB_DIR,
            collection_name=COLLECTION_NAME
        )
        # Explicitly persist the database to disk
        # db.persist()
        print(f"Created and persisted new database with {len(chunks)} documents")
    
    print("Vector database ready")
    return db

def initialize_gemini_llm():
    """Initialize the Google Gemini LLM."""
    print("Initializing Gemini model...")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

def create_qa_chain(llm, vector_db):
    """Create a question-answering chain."""
    print("Creating QA chain...")
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20}
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

def run_chatbot(qa_chain):
    """Run the interactive chatbot."""
    print("\n" + "="*50)
    print("Welcome to the PDF RAG Chatbot!")
    print("Ask questions about the content of your PDF.")
    print("Type 'exit' to quit the chatbot.")
    print("="*50 + "\n")
    
    while True:
        query = input("\nYour question: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("Thank you for using the PDF RAG Chatbot. Goodbye!")
            break
            
        try:
            print("\nSearching for answer...")
            result = qa_chain.invoke({"query": query})
            
            print("\nAnswer:")
            print(result["result"])
            
            # Uncomment to show source information
            # print("\nSources:")
            # for i, doc in enumerate(result["source_documents"]):
            #     print(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Sorry, I encountered an error while processing your question. Please try again.")

def main():
    """Main function to run the PDF RAG chatbot."""
    try:
        # Check if database already exists
        db_exists = os.path.exists(CHROMA_DB_DIR) and os.path.exists(os.path.join(CHROMA_DB_DIR, "chroma.sqlite3"))
        
        if not db_exists:
            print("No existing database found. Creating new database...")
            # Step 1: Extract text from PDF
            documents = extract_text_from_pdf(PDF_PATH)
            
            # Step 2: Chunk the text
            chunks = chunk_text(documents)
            
            # Step 3 & 4: Create vector database with embeddings
            vector_db = create_vector_database(chunks)
        else:
            print("Using existing vector database...")
            # Skip extraction and chunking, just load the database
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
            vector_db = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embedding_model,
                collection_name=COLLECTION_NAME
            )
        
        # Step 5: Initialize LLM
        llm = initialize_gemini_llm()
        
        # Create QA chain
        qa_chain = create_qa_chain(llm, vector_db)
        # Run interactive chatbot
        run_chatbot(qa_chain)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
