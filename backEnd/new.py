import fitz  # PyMuPDF
from google import genai

# Ensure the file path is correct and handle errors
file_path = "a.pdf"
client = genai.Client(api_key="AIzaSyBtunoQDSmYcWy1YiFGajaF3xJwR1NzjeA")
# Open the PDF and extract text from all pages
doc = fitz.open(file_path)
text = "".join(page.get_text() for page in doc)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=f"Consider this piece of text. {text}. Give me a brief summary of the text",
)
print(response.text)


