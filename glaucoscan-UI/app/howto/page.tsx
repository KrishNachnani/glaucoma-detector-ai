'use client';

export default function HowTo() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-[#0a192f] to-[#112240] py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Hero Section */}
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold text-white mb-2">How to Take a Fundus Image</h1>
          <p className="text-blue-300 text-lg">A Simple Guide to Capturing Eye Images for Glaucoma Screening</p>
        </div>

        {/* Video Tutorial */}
        <div className="mb-10">
          <h2 className="text-2xl font-semibold text-white mb-4">Watch Our Tutorial</h2>
          <div className="rounded-lg overflow-hidden shadow-lg" style={{ position: 'relative', paddingBottom: '56.25%' }}>
            <iframe 
              src="https://www.youtube.com/embed/iMut5iMbgIE?start=51" 
              title="How to Capture Fundus Image"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
              allowFullScreen
              className="w-full h-full"
              style={{ position: 'absolute', top: 0, left: 0 }}
            ></iframe>
          </div>
          <p className="text-sm text-gray-400 mt-2">
            Video courtesy of <a href="https://www.youtube.com/@JulianEspinosaOculoplastico" target="_blank" className="underline text-blue-400">Julián Espinosa @JulianEspinosaOculoplastico</a>
          </p>
        </div>

        {/* Instructions */}
        <div className="prose prose-invert max-w-none space-y-6 text-gray-300">
          <p>
            To use Glaucoscan.ai, you’ll need a clear retinal image (also called a fundus photo) of one eye. You can capture this using a smartphone and a 20D or 28D handheld lens. The image must show the optic nerve and surrounding retina.
          </p>

          <p>
            Hold the lens 2–4 cm in front of the patient’s eye, then align your smartphone flashlight through the lens until the retina is visible. You may need to adjust angles and distance to find the clearest view.
          </p>

          <p>
            Turn on your smartphone’s video mode (with flash), move slowly, and pause when the optic disc is in focus. Then take a screenshot of the clearest frame to upload for analysis.
          </p>

          <p>
            For best results, dim the room lights and ask the subject to keep both eyes open while fixating on a still object. Dilation is not required, but may improve clarity in darker eyes.
          </p>
        </div>
      </div>
    </div>
  );
} 
