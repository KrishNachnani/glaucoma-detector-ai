import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { Eye } from 'lucide-react';
import Link from 'next/link';
import NavBar from '@/components/NavBar'; 
import Script from "next/script";

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  metadataBase: new URL("https://glaucoscan.ai"),
  title: 'glaucoscan.ai - Glaucoma Detection',
  description: 'AI-powered glaucoma detection system', 
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-gradient-to-b from-[#0a192f] to-[#112240]`}>
        <NavBar />
        <Script
          id="founder-schema"
          type="application/ld+json"
          strategy="beforeInteractive"
        >
          {JSON.stringify({
            "@context": "https://schema.org",
            "@type": "Organization",
            name: "GlaucoScan.ai",
            url: "https://glaucoscan.ai",
            founder: {
              "@type": "Person",
              name: "Krish Nachnani",
              url: "https://glaucoscan.ai/story",
              sameAs: [
                "https://www.linkedin.com/in/krishnachnani/",
                "https://github.com/KrishNachnani",
                "https://scholar.google.com/citations?user=_sFaVUMAAAAJ&hl=en",
              ]
            }
          })}
        </Script>
        <div className="px-4 sm:px-6 lg:px-8">
          {children}
        </div>
      </body>
    </html>
  );
}