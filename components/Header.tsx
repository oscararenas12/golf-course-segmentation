'use client';

import Link from 'next/link';

export function Header() {
  return (
    <header className="h-[60px] bg-gray-900 border-b border-gray-800 flex items-center px-6">
      <Link
        href="/"
        className="flex items-center gap-3 hover:opacity-80 transition-opacity"
      >
        <span className="text-2xl">⛳</span>
        <h1 className="text-white">Golf Course Segmentation</h1>
      </Link>
    </header>
  );
}