import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { PasswordGate } from '@/components/PasswordGate';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Golf Course Segmentation',
  description: 'Analyze satellite imagery of golf courses using machine learning',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <PasswordGate>{children}</PasswordGate>
      </body>
    </html>
  );
}
