import { useState } from 'react'
import './App.css'
import prismLogo from './assets/prism.png' // Make sure to add your image

function App() {
  const port = 8000;
  const [messages, setMessages] = useState([])
  const [inputText, setInputText] = useState('')
  const [error, setError] = useState(null)
  const [selectedFile, setSelectedFile] = useState(null)

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (file && file.type === 'application/pdf') {
      setSelectedFile(file)
      
      // Create FormData to send file
      const formData = new FormData()
      formData.append('pdf', file)

      try {
        const response = await fetch(`http://127.0.0.1:${port}/upload-pdf`, {
          method: 'POST',
          body: formData,
        })
        
        const data = await response.json()
        if (data.success) {
          setMessages(prev => [...prev, { 
            text: `PDF uploaded successfully: ${file.name}`, 
            sender: 'system' 
          }])
        }
      } catch (error) {
        console.error('Error uploading PDF:', error)
        setError('Failed to upload PDF')
      }
    } else {
      setError('Please select a valid PDF file')
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    
    // Check if a PDF has been uploaded
    if (!selectedFile) {
      setError('Please upload a PDF file first')
      setMessages(prev => [...prev, { 
        text: 'Please upload a PDF file before asking questions', 
        sender: 'system' 
      }])
      return
    }
    
    try {
      // Add user message to chat
      const userMessage = { text: inputText, sender: 'user' }
      setMessages(prev => [...prev, userMessage])
      
      console.log('Sending to backend:', inputText)
      const response = await fetch(`http://127.0.0.1:${port}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputText }),
      })
      
      const data = await response.json()
      console.log('Received from backend:', data)

      if (data.error) {
        throw new Error(data.error)
      }
      
      // Add bot response to chat
      setInputText('')
      setMessages(prev => [...prev, { text: data.response, sender: 'bot' }])
    } catch (error) {
      console.error('Error:', error)
      setError(error.message)
      setMessages(prev => [...prev, { text: `Error: ${error.message}`, sender: 'error' }])
    }
  }

  return (
    <div className="chat-container">
      <div className="header">
        <img src={prismLogo} alt="PRISM Logo" />
        <h1>PRISM</h1>
      </div>

      <div className="file-upload">
        <input
          type="file"
          accept=".pdf"
          onChange={handleFileUpload}
          id="pdf-upload"
        />
        <label htmlFor="pdf-upload" className="upload-button">
          <svg className="upload-icon" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 20 20">
            <path d="M14.707 7.793a1 1 0 0 0-1.414 0L11 10.086V1.5a1 1 0 0 0-2 0v8.586L6.707 7.793a1 1 0 1 0-1.414 1.414l4 4a1 1 0 0 0 1.416 0l4-4a1 1 0 0 0-.002-1.414Z"/>
            <path d="M18 12h-2.55l-2.975 2.975a3.5 3.5 0 0 1-4.95 0L4.55 12H2a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-4a2 2 0 0 0-2-2Zm-3 5a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"/>
          </svg>
          Upload PDF
        </label>
        {selectedFile && (
          <span className="file-name">{selectedFile.name}</span>
        )}
      </div>
      
      <div className="messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.sender}`} style={{ whiteSpace: 'pre-wrap' }}>
          {message.text}
        </div>
        ))}
      </div>

      {error && <div className="error-message">{error}</div>}

      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Type your message..."
        />
        <button type="submit">Send</button>
      </form>
    </div>
  )
}

export default App
