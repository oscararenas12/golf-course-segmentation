'use client';

import { Mail, LogOut } from 'lucide-react';
import { Button } from './ui/button';

export function Footer() {
  const handleLogout = () => {
    localStorage.removeItem('golf-seg-auth');
    window.location.reload();
  };

  return (
    <footer className="bg-slate-900 border-t border-slate-800 px-6 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-slate-400 text-sm">
          <Mail className="size-4" />
          <span>Need access?</span>
          <a
            href="mailto:oscar.arenas01@student.csulb.edu"
            className="text-emerald-400 hover:text-emerald-300 underline"
          >
            oscar.arenas01@student.csulb.edu
          </a>
        </div>

        <div className="flex items-center gap-4">
          <span className="text-slate-500 text-xs">
            CSULB Research Project
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleLogout}
            className="text-slate-400 hover:text-white text-xs"
          >
            <LogOut className="size-3 mr-1" />
            Logout
          </Button>
        </div>
      </div>
    </footer>
  );
}
