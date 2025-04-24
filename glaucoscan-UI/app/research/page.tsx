'use client';

export default function ResearchPage() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-[#0a192f] to-[#112240] py-12 px-4">
      <div className="max-w-4xl mx-auto space-y-12">

        {/* Hero */}
        <div className="text-center">
          <h1 className="text-4xl font-bold text-white mb-2">Backed by Research</h1>
          <p className="text-blue-300 text-lg">Science That Powers Access.</p>
        </div>

        {/* Overview */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <p className="text-gray-300 text-lg">
            Our technology is based on cutting-edge research focused on creating lightweight AI models that outperform traditional heavy networks like ResNet on glaucoma diagnosis — while requiring far fewer resources.
          </p>
        </section>

        {/* Key Publications */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-4">Key Publications</h2>
          <ul className="text-gray-300 text-lg list-disc list-inside space-y-2">
            <li>
              <span className="font-medium text-white">Energy Efficient Learning Algorithms for Glaucoma Diagnosis</span> — Published in IEEE Xplore
              <br />
              <a href="https://doi.org/10.1109/ICMLA58977.2023.00307" target="_blank" className="text-blue-400 underline text-sm">https://doi.org/10.1109/ICMLA58977.2023.00307</a>
            </li>
            <li>
              <span className="font-medium text-white">GAN-based Data Augmentation for Advanced Glaucoma Diagnostics</span> — Featured in <i>Recent Advances in Deep Learning Applications</i>
              <br />
              <a href="https://www.taylorfrancis.com/books/edit/10.1201/9781003570882/recent-advances-deep-learning-applications-uche-onyekpe-vasile-palade-arif-wani" target="_blank" className="text-blue-400 underline text-sm">
                https://www.taylorfrancis.com/books/edit/10.1201/9781003570882
              </a>
            </li>
          </ul>
        </section>

        {/* Conferences */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-4">Presentations</h2>
          <p className="text-gray-300 text-lg">
            Our work has been presented at leading scientific conferences, including:
          </p>
          <ul className="text-gray-300 text-lg list-disc list-inside mt-2 space-y-1">
            <li>MIT Undergraduate Research Technology Conference</li>
            <li>IEEE ICMLA 2023</li>
          </ul>
        </section>
      </div>
    </main>
  );
}
