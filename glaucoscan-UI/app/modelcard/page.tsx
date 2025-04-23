import fs from 'fs';
import path from 'path';
import { compileMDX } from 'next-mdx-remote/rsc';

export default async function ModelCard() {
  // Read the MODEL_CARD.md file
  const modelCardPath = path.join(process.cwd(), 'MODEL_CARD.md');
  const modelCardContent = fs.readFileSync(modelCardPath, 'utf8');

  // Compile the MDX content
  const { content } = await compileMDX({
    source: modelCardContent,
    options: { parseFrontmatter: true }
  });

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#0a192f] to-[#112240] py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <article className="prose prose-invert prose-headings:text-white prose-a:text-blue-300 prose-strong:text-white prose-code:bg-gray-800 prose-code:p-1 prose-code:rounded max-w-none">
          {content}
        </article>
      </div>
    </div>
  );
}