from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
from google import genai
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

# Global variable to store current PDF context and chat history
current_context = ""
chat_history = ""

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    global current_context, chat_history
    
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract text from PDF
            doc = fitz.open(filepath)
            text = "".join(page.get_text() for page in doc)
            current_context = f"Consider this piece of text: {text}. Use this as context for the conversation."
            chat_history = ""  # Reset chat history when a new PDF is uploaded
            
            return jsonify({
                'success': True,
                'message': 'PDF uploaded and processed successfully'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    global current_context, chat_history
    
    data = request.json
    user_input = data.get('message')
    
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    if not current_context:
        return jsonify({'error': 'Please upload a PDF first'}), 400

    try:
        # Append user input to chat history
        chat_history += f"User: {user_input}\n"
        
        # Send user input to the Gemini API
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"{current_context}\n{chat_history}Assistant:",
        )
        
        # Append assistant response to chat history
        chat_history += f"Assistant: {response.text}\n"
        
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port =8000)