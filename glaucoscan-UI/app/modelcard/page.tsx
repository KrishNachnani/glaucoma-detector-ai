import fs from 'fs';
import path from 'path';
import { compileMDX } from 'next-mdx-remote/rsc';

export default async function ModelCard() {
  const modelCardPath = path.join(process.cwd(), 'MODEL_CARD.md');
  const modelCardContent = fs.readFileSync(modelCardPath, 'utf8');

  const { content } = await compileMDX({
    source: modelCardContent,
    options: { parseFrontmatter: true }
  });

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#0a192f] to-[#112240] py-12 px-4">
      <div className="max-w-4xl mx-auto">

        {/* Hero */}
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold text-white mb-2">Model Card</h1>
          <p className="text-blue-300 text-lg">Transparent Reporting for Responsible AI</p>
        </div>

        {/* Model Card Content */}
        <div className="bg-[#1a2942] p-6 rounded-xl">
          <article className="prose prose-invert prose-headings:text-white prose-a:text-blue-300 prose-strong:text-white prose-code:bg-gray-800 prose-code:p-1 prose-code:rounded max-w-none">
            {content}
          </article>
        </div>
      </div>
    </div>
  );
}
