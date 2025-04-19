from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
from google import genai

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Gemini client
client = genai.Client(api_key="AIzaSyBtunoQDSmYcWy1YiFGajaF3xJwR1NzjeA")

# Load PDF content
file_path = "a.pdf"
doc = fitz.open(file_path)
text = "".join(page.get_text() for page in doc)
context = f"Consider this piece of text: {text}. Use this as context for the conversation."

@app.route('/chat', methods=['POST'])
def chat():
    print("Received request") 
    data = request.json
    user_input = data.get('message')
    
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    try:
        # Send user input to the Gemini API
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"{context}\nUser: {user_input}\nAssistant:",
        )
        
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)