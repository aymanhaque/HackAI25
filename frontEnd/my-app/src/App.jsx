import { useState } from 'react'
import './App.css'

function App() {
  const [messages, setMessages] = useState([])
  const [inputText, setInputText] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    try {
      // Add user message to chat
      const userMessage = { text: inputText, sender: 'user' }
      setMessages([...messages, userMessage])
      
      // Send to backend
      const response = await fetch('http://127.0.0.1:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputText }),
      })
      
      const data = await response.json()
      
      // Add bot response to chat
      setMessages(prev => [...prev, { text: data.response, sender: 'bot' }])
      setInputText('')
    } catch (error) {
      console.error('Error:', error)
    }
  }

  return (
    <div className="chat-container">
      <h1>PRISM</h1>
      
      <div className="messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.sender}`}>
            {message.text}
          </div>
        ))}
      </div>

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
