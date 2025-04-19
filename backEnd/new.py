import fitz  # PyMuPDF

# Ensure the file path is correct and handle errors
file_path = "a.pdf"

try:
    # Open the PDF
    doc = fitz.open(file_path)

    # Extract text from all pages
    text = ""
    for page in doc:
        text += page.get_text()
    print(text[0:1000])
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
