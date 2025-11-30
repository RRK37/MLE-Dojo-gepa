import React from 'react';
import { Menu, Plus, Settings, User } from 'lucide-react';

export default function TopMenu() {
  return (
    <div className="top-menu">
      <div className="menu-left">
        <button className="icon-button" aria-label="Menu">
          <Menu size={20} />
        </button>
        <div className="model-selector">
          <span>GPT-4o</span>
        </div>
      </div>
      
      <div className="menu-right">
        <button className="new-chat-button">
          <Plus size={16} />
          <span>New Chat</span>
        </button>
        <button className="icon-button" aria-label="Settings">
          <Settings size={20} />
        </button>
        <button className="user-avatar" aria-label="User Profile">
          <User size={20} />
        </button>
      </div>
    </div>
  );
}
