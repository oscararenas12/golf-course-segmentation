'use client';

import { Github, BookOpen, HelpCircle } from 'lucide-react';
import { Button } from './ui/button';

interface HeaderProps {
  onLogoClick?: () => void;
}

export function Header({ onLogoClick }: HeaderProps) {
  return (
    <header className="h-[60px] bg-gray-900 border-b border-gray-800 flex items-center justify-between px-6">
      <button 
        onClick={onLogoClick}
        className="flex items-center gap-3 hover:opacity-80 transition-opacity"
      >
        <span className="text-2xl">⛳</span>
        <h1 className="text-white">Golf Course Segmentation</h1>
      </button>
      
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="sm" className="text-gray-300 hover:text-white">
          <BookOpen className="size-4 mr-2" />
          Docs
        </Button>
        <Button variant="ghost" size="sm" className="text-gray-300 hover:text-white">
          <Github className="size-4 mr-2" />
          GitHub
        </Button>
        <Button variant="ghost" size="sm" className="text-gray-300 hover:text-white">
          <HelpCircle className="size-4" />
        </Button>
      </div>
    </header>
  );
}