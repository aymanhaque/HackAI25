import { useState } from 'react'
import './App.css'

function App() {
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
        const response = await fetch('http://127.0.0.1:5000/upload-pdf', {
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
      const response = await fetch('http://127.0.0.1:5000/chat', {
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
      setMessages(prev => [...prev, { text: data.response, sender: 'bot' }])
      setInputText('')
    } catch (error) {
      console.error('Error:', error)
      setError(error.message)
      setMessages(prev => [...prev, { text: `Error: ${error.message}`, sender: 'error' }])
    }
  }

  return (
    <div className="chat-container">
      <h1>PRISM</h1>

      <div className="file-upload">
        <input
          type="file"
          accept=".pdf"
          onChange={handleFileUpload}
          id="pdf-upload"
        />
        <label htmlFor="pdf-upload" className="upload-button">
          Upload PDF
        </label>
        {selectedFile && (
          <span className="file-name">{selectedFile.name}</span>
        )}
      </div>
      
      <div className="messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.sender}`}>
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
