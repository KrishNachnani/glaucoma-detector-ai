'use client';

export default function About() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-[#0a192f] to-[#112240] py-12 px-4">
      <div className="max-w-4xl mx-auto space-y-12">
        
        {/* Hero Section */}
        <div className="text-center">
          <h1 className="text-4xl font-bold text-white mb-2">What We Do</h1>
          <p className="text-blue-300 text-lg">Making Early Detection Possible Anywhere.</p>
        </div>

        {/* About Section */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <p className="text-gray-300 text-lg">
            GlaucoScan.ai enables early glaucoma screening using only a smartphone, a handheld lens, and lightweight AI models. Our system is designed to detect signs of glaucoma without requiring expensive clinical infrastructure. By focusing on low cost hardware design, we’re making high-quality screening available in places where traditional tools are out of reach.
          </p>
        </section>

        {/* Mission */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-4">Our Mission</h2>
          <p className="text-gray-300 text-lg">
            We aim to close the diagnostic gap by delivering affordable, accessible screening tools to underserved communities around the world. Early detection should not depend on geography, income, or access to a specialist.
          </p>
        </section>

        {/* Technology */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-4">How It Works</h2>
          <p className="text-gray-300 text-lg">
            Users capture a video of the retina using their phone and 20 D lens attached to our low cost 3D printed device, extract the clearest still frame, and upload it to GlaucoScan.ai. Our neural networks detect optic nerve damage and provide interpretable visual outputs. The AI runs efficiently on mobile-compatible platforms, making it ideal for low-infrastructure settings.
          </p>
        </section>

        {/* Values */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-4">Open Collaboration</h2>
          <p className="text-gray-300 text-lg">
            We believe equity in healthcare requires openness. Our 3D tool and AI model is open source and designed to invite contributions from students, researchers, and clinicians. Together, we hope to expand the reach and reliability of glaucoma diagnostics for everyone.
          </p>
          <p className="text-gray-300 text-lg">
            You can access the source code for our glaucoma detection AI on GitHub :{' '}
            <a href="https://github.com/KrishNachnani/glaucoma-detector-ai" target="_blank" rel="noopener noreferrer" className="text-blue-500">
              Glaucoma Detector AI on GitHub
            </a>.
          </p>
          <p className="text-gray-300 text-lg">
            The code for the 3D printed tool is coming soon. Stay tuned for updates!
          </p>
          <p className="text-blue-400 text-sm mt-4 underline">
            <a href="/story">Read the story behind Glaucoscan.ai →</a>
          </p>
        </section>
      </div>
    </div>
  );
}
