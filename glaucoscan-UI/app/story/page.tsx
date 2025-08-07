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
              <h3 className="text-xl font-semibold text-white mb-2">Krish Nachnani</h3>
              <p>
                Krish Nachnani is a student researcher focused on advancing health equity through AI and low-cost medical tools. He founded GlaucoScan.ai to bring early glaucoma screening to communities that lack access to specialists. His work combines machine learning, smartphone-based imaging, and clinical validation in real-world settings.
              </p>
            </div>
          </div>
        </section>

        {/* Why I Built GlaucoScan.ai */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-4">Why I Built GlaucoScan.ai</h2>
          <p className="text-gray-300 text-lg">
            Living with progressive myopia and lattice degeneration, I’ve experienced the uncertainty of not knowing how my vision might change. That personal journey led me to study vision science and explore how technology could support earlier diagnosis for others.
          </p>
          <p className="text-gray-300 text-lg mt-4">
            While volunteering in Kenya, I saw how a lack of access to basic eye care often meant that glaucoma went undetected until vision loss had already occurred. In many communities, early screening simply wasn’t an option. I came back with a clear sense of urgency to build something practical that could help fill that gap.
          </p>
        </section>

        {/* A Low-Cost Screening Tool */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-4">A Low-Cost Screening Tool</h2>
          <p className="text-gray-300 text-lg">
            GlaucoScan.ai is designed to meet that need. It combines low-cost hardware with AI models to make early glaucoma detection more accessible. The hardware uses a 3D-printed smartphone adapter, a Volk 20D lens, and the phone’s built-in camera and flashlight. This setup allows for clear retinal imaging without the need for bulky or expensive equipment.
          </p>
        </section>

        {/* Field Testing in India */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-4">Field Testing in India</h2>
          <p className="text-gray-300 text-lg">
            To evaluate the tool in real-world conditions, I conducted pilot testing at two eye clinics in India: Ghaziabad Eye Hospital in Uttar Pradesh and Bhojay Sarvodaya Trust Clinic in Gujarat. More than 75 fundus images were captured and assessed for quality. These tests helped validate the system’s usability and confirmed that the hardware could reliably produce diagnostic-quality images in diverse clinical environments.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mt-6">
            <div className="w-full">
              <Image
                src="/images/MaleCheckup.jpg"
                alt="Fundus imaging using 3D-printed adapter at Ghazibad Eye Hospital"
                width={800}
                height={600}
                className="rounded-lg w-full h-auto object-cover"
              />
              <p className="text-gray-400 text-sm mt-2">
                Screening a male patient using the 3D-printed adapter
              </p>
            </div>
            <div className="w-full">
              <Image
                src="/images/FemaleCheckup.jpg"
                alt="Smartphone-based screening at Bhojay Sarvodaya Clinic"
                width={800}
                height={600}
                className="rounded-lg w-full h-auto object-cover"
              />
              <p className="text-gray-400 text-sm mt-2">
                Screening a female patient using the 3D-printed adapter
              </p>
            </div>
          </div>
        </section>

        {/* Improving the AI for Equity */}
        <section className="bg-[#1a2942] p-8 rounded-xl">
          <h2 className="text-2xl font-semibold text-white mb-4">Improving the AI for Equity</h2>
          <p className="text-gray-300 text-lg">
            The AI model behind GlaucoScan.ai has been continuously refined to ensure that it performs fairly across diverse populations. To improve equity across ethnicities, I modified the model using an approach called adversarial debiasing. This technique helps the AI focus on features relevant to glaucoma while minimizing the influence of demographic variables like race or ethnicity.
          </p>
          <p className="text-gray-300 text-lg mt-4">
            In addition to adversarial training, I’ve evaluated model performance across subgroups and iteratively updated datasets to support more balanced learning. This work is ongoing, with every version aimed at making the tool not only accurate but also equitable in how it serves patients from different backgrounds.
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
            <li>IEEE International Conference on Machine Learning and Applications (ICMLA) 2023</li>
          </ul>
        </section>
      </div>
    </main>
  );
}
