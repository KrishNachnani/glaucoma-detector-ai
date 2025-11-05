import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { Eye } from 'lucide-react';
import Link from 'next/link';
import NavBar from '@/components/NavBar'; 

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'glaucoscan.ai - Glaucoma Detection',
  description: 'AI-powered glaucoma detection system',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-gradient-to-b from-[#0a192f] to-[#112240]`}>
        <NavBar />
        <div className="px-4 sm:px-6 lg:px-8">
          {children}
        </div>
      </body>
    </html>
  );
}