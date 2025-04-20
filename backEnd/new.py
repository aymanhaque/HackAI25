import fitz  # PyMuPDF
from google import genai
import spacy
from collections import defaultdict
import networkx as nx
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import pytesseract
import io
import base64

client = genai.Client(api_key= "AIzaSyBnKfEVa42Dp-fgiBMTctsjESCiC0BjJYI")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "trust_remote_code": True
    },
    encode_kwargs={
        "batch_size": 64,
        "normalize_embeddings": True,
        "show_progress_bar": True
    },
    multi_process=True  # For parallel processing
)
clip_model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_pages = defaultdict(set)
        
    def build_from_docs(self, docs):
        """Extract entities and relationships from processed documents"""        
        for doc in docs:
             text = doc.page_content
             if not text.strip():
                continue
             page = doc.metadata.get('page', 0)
             spacy_doc = nlp(text)
            
            # Extract entities
             entities = [ent.text for ent in spacy_doc.ents]
            
            # Add nodes and track pages
             for ent in entities:
                self.graph.add_node(ent)
                self.entity_pages[ent].add(page)
                
            # Create relationships (co-occurrence in same sentence)
             for sent in spacy_doc.sents:
                sent_ents = [ent.text for ent in sent.ents]
                for i in range(len(sent_ents)):
                    for j in range(i+1, len(sent_ents)):
                        self.graph.add_edge(sent_ents[i], sent_ents[j], context=sent.text)

def process_pdf(file_path):
    # Extract with PyMuPDF4LLM (better table handling)
    loader = PyMuPDF4LLMLoader(
        file_path,
        table_strategy="lines",  # Changed from extract_tables
        mode="page",
        extract_images=True
    )
    docs = loader.load()
    
    # Enhanced image processing
    image_data = []
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for img_index, img in enumerate(page.get_images()):
            base_image = doc.extract_image(img[0])
            image_bytes = base_image["image"]
            
            # OCR with error handling
            try:
                pil_image = Image.open(io.BytesIO(image_bytes))
                ocr_text = pytesseract.image_to_string(pil_image)
            except Exception as e:
                ocr_text = f"OCR Error: {str(e)}"
            
            image_data.append({
                "page": page_num + 1,
                "image": base64.b64encode(image_bytes).decode(),
                "text": ocr_text,
                "type": "image"
            })
    
    return docs, image_data

def create_vector_stores(docs, image_data):
    # Text processing with modern splitter
    
    # Chroma with HuggingFace embeddings
    text_vectors = Chroma.from_documents(
        documents=docs,
        embedding= embeddings,
        persist_directory="./chroma_db"  # Added persistence
    )
    
    # Image processing pipeline
    image_embeddings = []
    image_metadata = []
    for img in image_data:
        # Use CLIP for image embeddings
        embedding = clip_model.encode(
            Image.open(io.BytesIO(base64.b64decode(img["image"])))
        )
        image_embeddings.append(embedding)
        image_metadata.append({
            "page": img["page"],
            "text": img["text"],
            "type": "image"
        })
    
    # Separate collection for images
    image_vectors = Chroma.from_embeddings(
        embeddings=image_embeddings,
        metadatas=image_metadata,
        embedding_function=clip_model.encode,
        collection_name="image_embeddings",
        persist_directory="./chroma_db"
    )
    
    return text_vectors, image_vectors

class GraphEnhancedRetriever:
    def __init__(self, text_vectors, image_vectors, knowledge_graph):
        self.text_vectors = text_vectors
        self.image_vectors = image_vectors
        self.knowledge_graph = knowledge_graph
        
    def query(self, input_text=None, top_k=5):
        results = []

        # Text search
        if input_text:
            text_results = self.text_vectors.similarity_search_with_relevance_scores(
                input_text, k=top_k
            )
            for doc, score in text_results:
                results.append({
                    "content": doc.page_content,
                    "source": f"Page {doc.metadata['page']}",
                    "type": "text",
                    "score": score
                })

        # Graph expansion
        query_entities = [ent.text for ent in nlp(input_text).ents]
        for entity in query_entities:
            if entity in self.knowledge_graph.graph:
                related = nx.single_source_shortest_path_length(
                    self.knowledge_graph.graph, entity, cutoff=2
                )
                for node, dist in related.items():
                    if dist > 0:
                        results.append({
                            "content": f"Related concept: {node} (distance {dist})",
                            "source": f"Pages {self.knowledge_graph.entity_pages.get(node, ['?'])}",
                            "type": "graph",
                            "score": 1 / (dist + 0.1)
                        })

        # Debugging output
        print("Query Entities:", query_entities)
        print("Retrieved Results:")
        for result in results:
            print(f"Source: {result['source']}, Score: {result['score']}, Content: {result['content'][:200]}")

        # Prioritize text results
        results = sorted(results, key=lambda x: (x["type"] != "text", -x["score"]))
        return results[:top_k]

# Open the PDF and extract text from all pages
try:
    docs, images = process_pdf("a.pdf")
    text_vectors, image_vectors = create_vector_stores(docs, images)

    kg = KnowledgeGraph()
    kg.build_from_docs(docs)


    retriever = GraphEnhancedRetriever(text_vectors, image_vectors, kg)
except Exception as e:
    print(f"Initialization Error: {str(e)}")
    exit(1)

# Split text into chunks
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# chunks = splitter.create_documents([text])

# # Generate embeddings
# model = SentenceTransformer("all-MiniLM-L6-v2")
# chunk_texts = [chunk.page_content for chunk in chunks]
# embeddings = model.encode(chunk_texts)

# # Store in FAISS index
# dimension = embeddings[0].shape[0]
# index = faiss.IndexFlatL2(dimension)
# index.add(np.array(embeddings))



# Initialize the chatbot loop
print("Chatbot initialized. Type 'exit' to quit.")
history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting chatbot. Goodbye!")
        break

    try:
        # Retrieve relevant context
        results = retriever.query(input_text=user_input, top_k=5)  # Use top 5 results instead of 3
        
        # Build context string
        context = "Document Context:\n"
        seen_sources = set()
        for result in results:
            if result['source'] not in seen_sources:
                context += f"- {result['source']}: {result['content'][:500]}...\n"
                seen_sources.add(result['source'])

        # Add image OCR texts if available
        if images:
            context += "\nImage Context:\n"
            for img in images[:2]:  # Show first 2 images' OCR
                context += f"- Page {img['page']} Image: {img['text'][:300]}...\n"

        # Format conversation history
        conversation_history = "\n".join(history[-4:])  # Keep last 2 exchanges
        
        # Generate response with context
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"""
            You are an intelligent assistant. Use the following document context and conversation history to answer the user's question in detail.

            Document Context:
            {context}

            Conversation History:
            {conversation_history}

            Question: {user_input}
            Provide a detailed and accurate answer based on the context above:
            """
        )
        
        # Handle response and update history
        assistant_response = response.text
        history.append(f"User: {user_input}")
        history.append(f"Assistant: {assistant_response}")
        
        # Print with sources
        print(f"Assistant: {assistant_response}")
        print(f"\nSources: {', '.join(seen_sources)}")

    except Exception as e:
        print(f"Error: {str(e)}")
        continue