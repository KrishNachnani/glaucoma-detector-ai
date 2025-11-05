"use client";

import Image from "next/image";
import Link from "next/link";

export default function AdapterPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-[#0a192f] to-[#112240] py-12 px-4">
      <div className="max-w-5xl mx-auto">
        {/* Hero Section */}
        <section className="text-center mb-14">
          <h1 className="text-4xl font-bold text-white mb-3">Glaucoscan 3D-Printed Adapter</h1>
          <p className="text-blue-200/80 max-w-2xl mx-auto text-base md:text-lg leading-relaxed">
            A lightweight, low-cost smartphone mount designed for handheld fundoscopy.
            This adapter aligns your phone’s camera and flash with a +20D lens to capture
            clear, coaxially illuminated retinal images for glaucoma screening.
          </p>

          
        </section>

        {/* Image Showcase */}
        <section className="grid md:grid-cols-2 gap-10 mb-14">
          <div className="relative rounded-xl overflow-hidden shadow-lg border border-blue-400/10">
            <Image
              src="/images/3DDevice.png"
              alt="Glaucoscan Adapter CAD Model"
              width={800}
              height={600}
              className="object-contain"
            />
          </div>
          <div className="relative rounded-xl overflow-hidden shadow-lg border border-blue-400/10">
            <Image
              src="/images/3DDeviceWithSmartphone.png"
              alt="Glaucoscan Adapter with Smartphone and +20D Lens"
              width={800}
              height={600}
              className="object-contain"
            />
          </div>
        </section>

        {/* Description */}
        <section className="bg-[#1a2942] p-8 rounded-xl mb-10">
          <h2 className="text-2xl font-semibold text-white mb-4">How It Works</h2>
          <p className="text-gray-300 text-lg mb-4">
            The Glaucoscan adapter maintains a fixed optical axis between the smartphone camera,
            flash, and the +20D lens, ensuring proper coaxial illumination of the retina. The
            mounting mechanism allows fine alignment to accommodate different phone models.
          </p>
          {/*<p className="text-gray-300 text-lg">
            The design can be printed on any consumer-grade 3D printer and assembled in minutes.
            It is made freely available for non-commercial use to help make retinal imaging more
            accessible in low-resource settings.
          </p>*/}
        </section>

        {/* Download Section */}
        <section className="bg-[#1a2942] p-8 rounded-xl mt-12">
          <div className="text-center">
            <h2 className="text-2xl font-semibold text-white mb-4">Download the Adapter File</h2>
            <p className="text-gray-300 text-lg mb-6 max-w-2xl mx-auto">
              You can 3D print your own Glaucoscan adapter using the provided G-code file.
              The model has been tested for compatibility with Volk +20D lenses and
              iPhone/Android smartphones. Printing takes approximately 5–6 hours using
              PLA at 0.2 mm layer height.
            </p>
            <Link
              href="/downloads/glaucoscan_adapter.gcode"
              target="_blank"
              className="inline-block bg-blue-500 hover:bg-blue-600 text-white font-semibold px-8 py-3 rounded-lg transition shadow-lg"
            >
              ⬇️ Download G-code
            </Link>
            <p className="text-xs text-gray-400 mt-3">
              Free for non-commercial use under a Creative Commons BY-NC 4.0 license.
            </p>
          </div>
        </section>

      </div>
    </div>
  );
}
