'use client';

import React, { useState } from 'react';
import TopMenu from '@/components/TopMenu';
import ChatArea from '@/components/ChatArea';
import InputArea from '@/components/InputArea';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);

  const handleSendMessage = (content: string) => {
    const newUserMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
    };

    setMessages((prev) => [...prev, newUserMessage]);

    // Mock AI response
    setTimeout(() => {
      const newAiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `I am a mock AI. You said: "${content}". This is a demonstration of the frontend interface.`,
      };
      setMessages((prev) => [...prev, newAiMessage]);
    }, 1000);
  };

  return (
    <div className="app-container">
      <TopMenu />
      <ChatArea messages={messages} />
      <InputArea onSendMessage={handleSendMessage} />
    </div>
  );
}
