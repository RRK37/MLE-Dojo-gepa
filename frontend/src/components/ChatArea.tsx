import React from 'react';
import { Bot, User } from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

interface ChatAreaProps {
  messages: Message[];
}

export default function ChatArea({ messages }: ChatAreaProps) {
  return (
    <div className="chat-area">
      {messages.length === 0 ? (
        <div className="empty-state">
          <div className="logo-placeholder">
            <Bot size={48} />
          </div>
          <h2>How can I help you today?</h2>
        </div>
      ) : (
        <div className="messages-container">
          {messages.map((msg) => (
            <div key={msg.id} className={`message-row ${msg.role}`}>
              <div className="message-content">
                <div className="avatar">
                  {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
                </div>
                <div className="text">
                  {msg.content}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
