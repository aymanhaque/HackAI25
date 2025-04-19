import fitz  # PyMuPDF
from google import genai

# Ensure the file path is correct and handle errors
file_path = "a.pdf"
client = genai.Client(api_key="AIzaSyBtunoQDSmYcWy1YiFGajaF3xJwR1NzjeA")

# Open the PDF and extract text from all pages
doc = fitz.open(file_path)
text = "".join(page.get_text() for page in doc)

# Initialize the chatbot loop
print("Chatbot initialized. Type 'exit' to quit.")
context = f"Consider this piece of text: {text}. Use this as context for the conversation."

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting chatbot. Goodbye!")
        break

    # Send user input to the Gemini API
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{context}\nUser: {user_input}\nAssistant:",
    )
    print(f"Assistant: {response.text}")