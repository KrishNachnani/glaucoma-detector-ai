"use client";

import { useState } from "react";
import Link from "next/link";
import { Eye } from "lucide-react";

export default function NavBar() {
  const [open, setOpen] = useState(false);

  return (
    <nav className="bg-[#0a192f] p-6 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        <Link href="/" className="flex items-center gap-2">
          <Eye className="w-8 h-8 text-blue-400" />
          <span className="text-2xl font-bold text-white">glaucoscan.ai</span>
        </Link>

        <div className="flex flex-wrap gap-4 justify-end text-sm">
          <Link href="/about" className="text-gray-300 hover:text-white transition-colors">About</Link>
          <Link href="/contact" className="text-gray-300 hover:text-white transition-colors">Contact</Link>
          <Link href="/story" className="text-gray-300 hover:text-white transition-colors">Our Story</Link>

          <div
            className="relative"
            onMouseEnter={() => setOpen(true)}
            onMouseLeave={() => setOpen(false)}
          >
            <button
              className="text-gray-300 hover:text-white transition-colors"
              onClick={() => setOpen(v => !v)}
              aria-haspopup="menu"
              aria-expanded={open}
            >
              Using Glaucoscan
            </button>
            {open && (
              <div
                role="menu"
                className="absolute left-0 top-full bg-[#112240] border border-blue-400/20 rounded-lg py-2 w-52 shadow-lg"
              >
                <Link href="/capture" role="menuitem" className="block px-4 py-2 text-sm text-gray-300 hover:bg-blue-900/40 hover:text-white transition">
                  Capture Retinal Images
                </Link>
                <Link href="/adapter" role="menuitem" className="block px-4 py-2 text-sm text-gray-300 hover:bg-blue-900/40 hover:text-white transition">
                  Download the Adapter
                </Link>
              </div>
            )}
          </div>

          <Link href="/modelcard" className="text-gray-300 hover:text-white transition-colors">Model Card</Link>
        </div>
      </div>
    </nav>
  );
}
