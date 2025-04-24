'use client';

import Image from 'next/image';

export default function StoryPage() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-[#0a192f] to-[#112240] py-12 px-4">
      <div className="max-w-4xl mx-auto space-y-12">

        {/* Our Story Title */}
        <div className="text-center">
          <h1 className="text-4xl font-bold text-white mb-2">Our Story</h1>
          <p className="text-blue-300 text-lg">Built from Experience. Designed for Impact.</p>
        </div>

        {/* About Me Section */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-6">About the Founder</h2>
          <div className="flex flex-col md:flex-row items-center md:items-start gap-6">
            <div className="w-36 h-36 rounded-full overflow-hidden border-2 border-blue-500 flex-shrink-0">
              <Image
                src="/images/founder.jpg"
                alt="Krish Nachnani"
                width={144}
                height={144}
                className="object-cover w-full h-full"
              />
            </div>
            <div className="text-gray-300 text-lg">
              <h3 className="text-xl font-semibold text-white mb-2">I'm Krish Nachnani</h3>
              <p>
                I'm a researcher passionate about using technology to bridge healthcare gaps. My work focuses on creating
                energy-efficient, AI-driven tools that expand access to critical diagnostics — especially in communities where traditional resources are limited.
              </p>
            </div>
          </div>
        </section>

        {/* Why This Matters */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-4">Why This Matters</h2>
          <p className="text-gray-300 text-lg">
            Living with progressive myopia and lattice degeneration, I’ve spent years navigating the anxiety of how quickly my vision could change — and what that might mean for my future. It's what initially drew me to explore the science of vision, and eventually, to build tools that might help others in the same position.
          </p>
          <p className="text-gray-300 text-lg mt-4">
            During a volunteer trip to Kenya, I saw how limited access to eye care turns uncertainty into inevitability — where many don’t even know they have glaucoma until it’s too late. It was a wake-up call. I realized that the same concerns I faced with my own vision were magnified many times over in places without the tools or specialists to intervene early.
          </p>
          <p className="text-gray-300 text-lg mt-4">
            Glaucoscan.ai was built in response to that gap — not as a research showcase, but as a practical tool that could empower communities, clinics, and schools to catch glaucoma early, affordably, and accurately.
          </p>
        </section>

        {/* A Silent Threat */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-4">A Silent Threat</h2>
          <p className="text-gray-300 text-lg">
            Glaucoma affects over 76 million people globally — yet most people don’t even know they have it until vision is permanently lost. It disproportionately affects communities of African descent and rural regions, where access to ophthalmologists is scarce. In many places, there is less than one ophthalmologist per 100,000 people.
          </p>
        </section>

        {/* Building GlaucoScan.ai */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-4">Building GlaucoScan.ai</h2>
          <p className="text-gray-300 text-lg">
            This tool began as a science fair project — but quickly evolved into a research-backed system designed for real-world impact. From the start, I knew it had to work under constraints: low compute power, low internet access, high urgency.
          </p>
          <p className="text-gray-300 text-lg mt-4">
            By leveraging lightweight models like MLPs and enhancing data diversity through GAN-based augmentation, I built an AI pipeline that could detect glaucoma from fundus images using just a smartphone and a basic lens adapter. Fast, low-cost, and accessible — without compromising accuracy.
          </p>
        </section>

        {/* The Vision */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-4">The Vision</h2>
          <p className="text-gray-300 text-lg">
            The mission behind Glaucoscan.ai is to democratize access to vision-saving technologies. We want to ensure that early glaucoma detection tools are not just available to the privileged few, but to schools, mobile clinics, and rural communities around the world.
          </p>
          <p className="text-gray-300 text-lg mt-4">
            By open-sourcing the platform and inviting students, researchers, and clinicians to contribute, we hope to create a growing ecosystem of tools that put diagnostic power directly in the hands of communities who need it most.
          </p>
        </section>
      </div>

      {/* Research Teaser Section */}
      <div className="max-w-4xl mx-auto px-4 mt-12">
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-4">Backed by Research</h2>
          <p className="text-gray-300 text-lg mb-4">
            Glaucoscan.ai is powered by peer-reviewed studies and real-world experiments. Our lightweight AI models are not only energy efficient — they’re built to outperform traditional models like ResNet in diagnosing glaucoma from retinal images.
          </p>
          <p className="text-blue-400 text-sm underline">
            <a href="/research">Explore the research behind Glaucoscan.ai →</a>
          </p>
        </section>
      </div>
    </main>
  );
}
