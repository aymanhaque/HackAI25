import fitz  # PyMuPDF
from google import genai
import spacy
import networkx as nx
from collections import defaultdict
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders.parsers import TesseractBlobParser
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import pytesseract
import io
import base64
import os
import numpy as np
from functools import lru_cache
import hashlib
import pickle
from langchain.schema import Document
from chromadb.utils import embedding_functions
from fuzzywuzzy import fuzz
import camelot
import re
import traceback

MAX_CONTEXT_LENGTH = 2000  # Adjust based on your LLM's token limit

def truncate_context(context):
    """Truncate context to fit within the token limit."""
    total_length = sum(len(c) for c in context)
    while total_length > MAX_CONTEXT_LENGTH and context:
        context.pop(0)  # Remove the oldest context entry
        total_length = sum(len(c) for c in context)
    return context

print("Current working directory:", os.getcwd())

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
docs = images = text_store = image_store = kg = retriever = None
# Initialize Gemini client
client = genai.Client(api_key="AIzaSyBnKfEVa42Dp-fgiBMTctsjESCiC0BjJYI")
clip_model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')

# Handle spaCy model installation
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def process_pdf(file_path, pages_to_process=None):
    """Process the PDF to extract text, images, and tables with improved extraction."""

    # Initialize the PyMuPDF4LLMLoader with appropriate configurations
    loader = PyMuPDF4LLMLoader(
        file_path,
        mode="page",  # Extract content by page
        extract_images=True,  # Enable image extraction
        images_parser=TesseractBlobParser(langs=["eng"]),  # Use Tesseract for OCR
        table_strategy="lines",  # Use "lines" strategy for table extraction
    )

    # Load the documents
    docs = loader.load()

    # Filter pages if specific pages are requested
    if pages_to_process:
        docs = [doc for doc in docs if doc.metadata.get("page", 0) + 1 in pages_to_process]

    # Extract raw page content for backup retrieval
    raw_page_content = {}
    for doc in docs:
        page_num = doc.metadata.get("page", 0)
        raw_page_content[page_num] = {
            "text": doc.page_content,
            "metadata": doc.metadata,
        }

    # Extract images from the documents
    image_data = []
    for doc in docs:
        page_num = doc.metadata.get("page", 0)
        if "images" in doc.metadata:
            for img in doc.metadata["images"]:
                image_data.append({
                    "page": page_num + 1,
                    "image": img.get("image", ""),
                    "text": img.get("text", ""),
                    "type": "image",
                })

    # Extract tables from the documents
    tables_data = []
    for doc in docs:
        page_num = doc.metadata.get("page", 0)
        if "tables" in doc.metadata:
            for table in doc.metadata["tables"]:
                tables_data.append({
                    "page": page_num + 1,
                    "data": table.get("data", ""),
                    "type": "table",
                })

    return docs, image_data, tables_data, raw_page_content

# Helper function to extract financial data from text
def extract_financial_data(text):
    """Extract financial figures using patterns common in financial reports."""
    import re
    
    # Pattern for currency amounts with commas and optional decimals
    # Examples: INR 355,170 Million, $12.5 Billion, ₹1,234.56 Crore
    currency_pattern = r'(?:(?:INR|Rs\.|\$|₹)\s?)([\d,]+(?:\.\d+)?)\s?(?:Million|Billion|Crore|Lakh|Mn|Bn)'
    
    # Extract all matches
    matches = re.findall(currency_pattern, text)
    
    # Clean and convert the matches to standard format
    financial_data = {}
    for i, match in enumerate(re.finditer(currency_pattern, text)):
        full_match = match.group(0)
        value = match.group(1)
        # Get surrounding context (30 chars before and after)
        start = max(0, match.start() - 30)
        end = min(len(text), match.end() + 30)
        context = text[start:end]
        
        # Try to identify what this figure represents
        figure_type = "unknown"
        context_lower = context.lower()
        if "revenue" in context_lower:
            figure_type = "revenue"
        elif "profit" in context_lower or "pat" in context_lower:
            figure_type = "profit"
        elif "ebitda" in context_lower:
            figure_type = "ebitda"
        elif "asset" in context_lower:
            figure_type = "asset"
        
        financial_data[f"{figure_type}_{i}"] = {
            "value": value,
            "full_text": full_match,
            "context": context,
            "type": figure_type
        }
    
    return financial_data

def create_vector_stores(docs, image_data):
    """Create vector stores for text and images."""
    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)
    text_store_file = os.path.join(cache_dir, "text_store_dir.pkl")
    image_store_file = os.path.join(cache_dir, "image_store_dir.pkl")

    if os.path.exists(text_store_file) and os.path.exists(image_store_file):
        print("Loading cached vector store directories...")
        with open(text_store_file, "rb") as f:
            text_store_dir = pickle.load(f)
        with open(image_store_file, "rb") as f:
            image_store_dir = pickle.load(f)

        # Reconstruct the vector stores from their persist directories
        text_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        text_store = Chroma(
            persist_directory=text_store_dir,
            embedding_function=text_embeddings
        )

        image_store = None
        if image_store_dir:
            image_embeddings = HuggingFaceEmbeddings(
                model_name="clip-ViT-B-32-multilingual-v1"
            )
            image_store = Chroma(
                persist_directory=image_store_dir,
                embedding_function=image_embeddings
            )

        return text_store, image_store
    print("Creating vector stores...")
    text_store_dir = "./chroma_db"
    text_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    text_store = Chroma.from_documents(
        docs,
        embedding=text_embeddings,
        persist_directory=text_store_dir
    )

    image_store = None
    if image_data:
        image_store_dir = "./chroma_db_images"
        image_embeddings = HuggingFaceEmbeddings(
            model_name="clip-ViT-B-32-multilingual-v1"
        )
        image_documents = []
        for img in image_data:
            image = Image.open(io.BytesIO(base64.b64decode(img["image"]))).convert("RGB")
            image_documents.append(
                Document(
                    page_content="",
                    metadata={
                        "page": img["page"],
                        "text": img["text"],
                        "type": "image"
                    }
                )
            )
        image_store = Chroma.from_documents(
            image_documents,
            embedding=image_embeddings,
            persist_directory=image_store_dir
        )

    print(f"Text Store Directory: {text_store_dir}")
    if image_store:
        print(f"Image Store Directory: {image_store_dir}")

    # Save only the persist directories
    with open(text_store_file, "wb") as f:
        pickle.dump(text_store_dir, f)
    with open(image_store_file, "wb") as f:
        pickle.dump(image_store_dir if image_store else None, f)

    return text_store, image_store

class EnhancedRetriever:
    def __init__(self, text_store, image_store=None, raw_page_content=None):
        self.text_store = text_store
        self.image_store = image_store
        self.raw_page_content = raw_page_content or {}

    def query(self, query, top_k=5):
        """Retrieve relevant documents and images."""
        results = []

        # Retrieve relevant text documents
        vector_results = self.text_store.similarity_search_with_relevance_scores(query, k=top_k)
        for doc, score in vector_results:
            results.append({
                "content": doc.page_content,
                "page": doc.metadata.get('page', '?'),
                "type": "text",
                "score": score,
                "source": f"Page {doc.metadata.get('page', '?')}"
            })

        # Retrieve relevant images if the query mentions "image" or "diagram"
        if self.image_store and ("image" in query.lower() or "diagram" in query.lower()):
            image_results = self.image_store.similarity_search_with_relevance_scores(query, k=top_k)
            for doc, score in image_results:
                results.append({
                    "content": doc.metadata.get("text", ""),  # Associated text with the image
                    "page": doc.metadata.get("page", '?'),
                    "type": "image",
                    "score": score,
                    "source": f"Page {doc.metadata.get('page', '?')}"
                })

        # Sort results by relevance score
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]

def build_context(results):
    """Build a concise context for the LLM."""
    context = ["Document Context:"]
    for res in results:
        if res["type"] == "text":
            context.append(f"- Page {res['page']}: {res['content'][:500]}...")
        elif res["type"] == "image":
            context.append(f"- Image (Page {res['page']}): {res['content']}")
    return "\n".join(context)

def get_file_hash(file_path):
    """Generate a SHA-256 hash for the given file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def is_cache_valid(file_path, cache_file):
    """Check if the cache is valid by comparing file hashes."""
    if not os.path.exists(cache_file):
        return False
    cached_hash = get_file_hash(file_path)
    with open(cache_file, "rb") as f:
        cached_data = pickle.load(f)
    return cached_data.get("file_hash") == cached_hash

def save_cache(data, cache_file):
    """Save processed data to a cache file."""
    # Exclude non-pickleable objects
    data_to_save = {
        "docs": data["docs"],
        "images": data["images"],
        "text_store_data": "./chroma_db",  # Explicitly store the text store directory
        "image_store_data": "./chroma_db_images" if data["image_store"] else None,
        "knowledge_graph_data": {
            "graph": nx.node_link_data(data["knowledge_graph"].graph),
            "entity_pages": data["knowledge_graph"].entity_pages
        }
    }
    with open(cache_file, "wb") as f:
        pickle.dump(data_to_save, f)

def load_cache(cache_file):
    """Load processed data from a cache file."""
    with open(cache_file, "rb") as f:
        data = pickle.load(f)

    # Reconstruct non-pickleable objects
    text_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    text_store = Chroma(
        persist_directory=data["text_store_data"],
        embedding_function=text_embeddings
    ) if data["text_store_data"] else None

    image_store = None
    if data["image_store_data"]:
        image_embeddings = HuggingFaceEmbeddings(
            model_name="clip-ViT-B-32-multilingual-v1"
        )
        image_store = Chroma(
            persist_directory=data["image_store_data"],
            embedding_function=image_embeddings
        )

    kg = KnowledgeGraph()
    kg.graph = nx.node_link_graph(data["knowledge_graph_data"]["graph"])
    kg.entity_pages = data["knowledge_graph_data"]["entity_pages"]

    return {
        "docs": data["docs"],
        "images": data["images"],
        "text_store": text_store,
        "image_store": image_store,
        "knowledge_graph": kg
    }

@lru_cache(maxsize=1)
def get_processed_data(file_path):
    """Process the PDF or load cached data if available."""
    file_hash = get_file_hash(file_path)
    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{file_hash}_full.pkl")

    if is_cache_valid(file_path, cache_file):
        print("Loading cached data...")
        return load_cache(cache_file)

    print("Processing document...")
    # Unpack all four returned values
    docs, images, tables, raw_page_content = process_pdf(file_path)
    text_store, image_store = create_vector_stores(docs, images)

    data = {
        "docs": docs,
        "images": images,
        "tables": tables,  # Include tables in the processed data
        "raw_page_content": raw_page_content,  # Include raw page content
        "text_store": text_store,
        "image_store": image_store,
        "knowledge_graph": kg
    }
    save_cache(data, cache_file)
    return data

def extract_relevant_sentence(content, query):
    """Extract a relevant sentence from the content based on the query."""
    sentences = content.split('.')
    best_match = None
    highest_score = 0
    for sentence in sentences:
        score = fuzz.partial_ratio(query.lower(), sentence.lower())
        if score > highest_score:
            highest_score = score
            best_match = sentence.strip()
    if best_match and highest_score > 70:  # Set a threshold for relevance
        print(f"Matching Sentence Found: {best_match} (Score: {highest_score})")
        return best_match
    print("No matching sentence found.")
    return None

# Initialize system
try:
    file_path = r"c:\Users\magic\OneDrive\Documents\HackAI25\backEnd\a.pdf"
    processed_data = get_processed_data(file_path)
    docs = processed_data["docs"]
    text_store = processed_data["text_store"]
    retriever = EnhancedRetriever(text_store, raw_page_content=processed_data["raw_page_content"])
except Exception as e:
    print(f"Initialization Error: {str(e)}")
    docs = text_store = retriever = None

print("Processed Documents:")
for doc in docs[:3]:  # Print the first 3 documents
    print(f"Page {doc.metadata.get('page', '?')}: {doc.page_content[:500]}")

# Example query for diagram related to revenue trends
user_input = "Show me the diagram related to revenue trends."
results = retriever.query(user_input, top_k=5)
formatted_context = build_context(results)
print(formatted_context)

print("Chatbot initialized. Type 'exit' to quit.")
history = []

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Retrieve context using the retriever
        results = retriever.query(user_input, top_k=5)
        formatted_context = build_context(results)

        # Generate response using the LLM
        prompt = f"""
        You are an intelligent assistant specialized in financial document analysis.
        Use the following document context to answer the user's question:

        Document Context:
        {formatted_context}

        Question: {user_input}

        Provide a detailed and accurate answer based on the context above.
        """
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        print(f"Assistant: {response.text}")

        # Update conversation history
        history.append(f"User: {user_input}")
        history.append(f"Assistant: {response.text}")

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()