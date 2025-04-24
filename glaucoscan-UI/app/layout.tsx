import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { Eye } from 'lucide-react';
import Link from 'next/link';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'glaucoscan.ai - Glaucoma Detection',
  description: 'AI-powered glaucoma detection system',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-gradient-to-b from-[#0a192f] to-[#112240]`}>
        <nav className="bg-[#0a192f] p-6 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto flex justify-between items-center">
            <Link href="/" className="flex items-center gap-2">
              <Eye className="w-8 h-8 text-blue-400" />
              <span className="text-2xl font-bold text-white">glaucoscan.ai</span>
            </Link>

            <div className="flex flex-wrap gap-4 justify-end text-sm">
              <Link href="/about" className="text-gray-300 hover:text-white transition-colors">
                About
              </Link>
              <Link href="/contact" className="text-gray-300 hover:text-white transition-colors">
                Contact
              </Link>
              <Link href="/story" className="text-gray-300 hover:text-white transition-colors">
                Our Story
              </Link>
              <Link href="/howto" className="text-gray-300 hover:text-white transition-colors">
                How to Use
              </Link>
              <Link href="/modelcard" className="text-gray-300 hover:text-white transition-colors">
                Model Card
              </Link>
            </div>
          </div>
        </nav>

        <div className="px-4 sm:px-6 lg:px-8">
          {children}
        </div>
      </body>
    </html>
  );
}