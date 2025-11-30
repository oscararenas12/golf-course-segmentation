'use client';

import { useState, useEffect } from 'react';
import { Lock, Mail } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';

interface PasswordGateProps {
  children: React.ReactNode;
}

// Simple hash function for password verification (not cryptographically secure, but fine for demo protection)
const hashPassword = (password: string): string => {
  let hash = 0;
  for (let i = 0; i < password.length; i++) {
    const char = password.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return hash.toString();
};

// The hashed password - change this by running: hashPassword('your-password')
// Current password: "golfcourse2025"
const VALID_PASSWORD_HASH = '-1544aborrar42';

export function PasswordGate({ children }: PasswordGateProps) {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Check if user is already authenticated
    const authStatus = localStorage.getItem('golf-seg-auth');
    if (authStatus === 'authenticated') {
      setIsAuthenticated(true);
    } else {
      setIsAuthenticated(false);
    }
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    // Simple password check
    setTimeout(() => {
      if (password === 'golfcourse2025') {
        localStorage.setItem('golf-seg-auth', 'authenticated');
        setIsAuthenticated(true);
      } else {
        setError('Incorrect password. Please try again or request access.');
      }
      setIsLoading(false);
    }, 500);
  };

  // Show loading state while checking auth
  if (isAuthenticated === null) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  // Show password form if not authenticated
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-4">
        <div className="w-full max-w-md">
          {/* Logo/Header */}
          <div className="text-center mb-8">
            <div className="text-6xl mb-4">⛳</div>
            <h1 className="text-2xl font-bold text-white mb-2">
              Golf Course Segmentation
            </h1>
            <p className="text-slate-400">
              AI-powered semantic segmentation for golf course analysis
            </p>
          </div>

          {/* Password Form */}
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <Lock className="size-5 text-emerald-500" />
              <h2 className="text-lg font-semibold text-white">Access Required</h2>
            </div>

            <p className="text-slate-400 text-sm mb-6">
              This application is password protected. Enter the access code to continue.
            </p>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <Input
                  type="password"
                  placeholder="Enter password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="bg-slate-700 border-slate-600 text-white placeholder:text-slate-500"
                  autoFocus
                />
                {error && (
                  <p className="text-red-400 text-sm mt-2">{error}</p>
                )}
              </div>

              <Button
                type="submit"
                disabled={isLoading || !password}
                className="w-full bg-emerald-600 hover:bg-emerald-700"
              >
                {isLoading ? (
                  <div className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white"></div>
                    Verifying...
                  </div>
                ) : (
                  'Access Application'
                )}
              </Button>
            </form>
          </div>

          {/* Request Access */}
          <div className="mt-6 p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
            <div className="flex items-start gap-3">
              <Mail className="size-5 text-slate-400 mt-0.5" />
              <div>
                <p className="text-slate-300 text-sm font-medium">
                  Need access?
                </p>
                <p className="text-slate-400 text-sm mt-1">
                  Please email{' '}
                  <a
                    href="mailto:oscar.arenas01@student.csulb.edu"
                    className="text-emerald-400 hover:text-emerald-300 underline"
                  >
                    oscar.arenas01@student.csulb.edu
                  </a>
                  {' '}to request access credentials.
                </p>
              </div>
            </div>
          </div>

          {/* Footer */}
          <p className="text-center text-slate-500 text-xs mt-6">
            California State University, Long Beach
          </p>
        </div>
      </div>
    );
  }

  // User is authenticated, show the app
  return <>{children}</>;
}
