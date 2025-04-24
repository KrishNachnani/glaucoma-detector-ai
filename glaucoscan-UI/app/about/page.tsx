'use client';

export default function About() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-[#0a192f] to-[#112240] py-12 px-4">
      <div className="max-w-4xl mx-auto space-y-12">
        
        {/* Hero Section */}
        <div className="text-center">
          <h1 className="text-4xl font-bold text-white mb-2">What We Do</h1>
          <p className="text-blue-300 text-lg">Making Early Detection Possible — Anywhere.</p>
        </div>

        {/* Mission */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-4">Our Mission</h2>
          <p className="text-gray-300 text-lg">
            Glaucoscan.ai is a global initiative to democratize access to glaucoma diagnostics. We believe that early detection shouldn’t depend on location, income, or access to specialists. Our mission is to make vision-saving tools available to everyone — especially in the communities that need them most.
          </p>
        </section>

        {/* Technology */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-4">How It Works</h2>
          <p className="text-gray-300 text-lg">
            Using lightweight neural networks and smartphone fundus photography, Glaucoscan.ai can screen for signs of glaucoma in seconds. Our energy-efficient models are trained for speed and accessibility — so a device as simple as a phone paired with a lens can help catch a silent cause of blindness before it’s too late.
          </p>
        </section>

        {/* Values */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-4">What We Stand For</h2>
          <p className="text-gray-300 text-lg">
            We’re committed to equity, transparency, and collaboration. We build tools that work in real-world conditions — low internet, low power, low cost — and we believe in open research that invites others to improve, replicate, or expand our impact.
          </p>
          <p className="text-blue-400 text-sm mt-4 underline">
            <a href="/story">Read the story behind Glaucoscan.ai →</a>
          </p>
        </section>
      </div>
    </div>
  );
}
