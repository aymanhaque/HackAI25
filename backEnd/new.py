import fitz  # PyMuPDF
from google import genai
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np

# Ensure the file path is correct and handle errors
file_path = "a.pdf"
client = genai.Client(api_key="AIzaSyBtunoQDSmYcWy1YiFGajaF3xJwR1NzjeA")

# Open the PDF and extract text from all pages
doc = fitz.open(file_path)
text = "".join(page.get_text() for page in doc)

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
history = [f"Consider this piece of text: {text}. Use this as context for the conversation."]

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting chatbot. Goodbye!")
        break

    # Add user input to the history
    history.append(f"User: {user_input}")

    # Prepare the conversation context by joining the history
    conversation_context = "\n".join(history)

    # Send the conversation context to the Gemini API
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{conversation_context}\nAssistant:",
    )

    # Add the assistant's response to the history
    assistant_response = response.text
    history.append(f"Assistant: {assistant_response}")

    # Print the assistant's response
    print(f"Assistant: {assistant_response}")