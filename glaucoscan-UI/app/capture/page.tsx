"use client";

import { useState } from "react";
import Link from "next/link";

export default function CaptureRetinalImages() {
  // 🔧 Replace this with your own hosted video URL when ready
  const CURRENT_TUTORIAL_EMBED =
    "https://www.youtube.com/embed/iMut5iMbgIE?start=51";

  const [openSteps, setOpenSteps] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#0a192f] to-[#112240] py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Hero Section */}
        <section className="text-center mb-14">
          <h1 className="text-4xl font-bold text-white mb-3">Capture Retinal Images</h1>
          <div className="max-w-2xl mx-auto text-blue-200/80 text-base md:text-lg leading-relaxed">
            A guide to taking fundus photos using your smartphone and a +20D lens
          </div>

        </section>

        {/* Introductory Text and Coming Soon Notice */}
        <section className="bg-[#1a2942] p-8 rounded-xl mb-10">
          <div className="space-y-5 text-gray-200 leading-relaxed">
            <p className="text-lg text-blue-100/90">
              This tutorial demonstrates proper alignment and lighting to obtain
              fundus photographs suitable for AI screening.
            </p>

            <div className="bg-[#112B4A] text-blue-100 rounded-xl p-5 shadow-inner flex items-start gap-3">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
                strokeWidth={1.6} stroke="currentColor"
                className="w-6 h-6 text-blue-300 mt-0.5 flex-shrink-0">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M12 9v3.75m0 3.75h.008v.008H12V16.5zm9 0a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-sm md:text-base">
                <span className="font-semibold text-blue-200">Coming soon:</span> our own tutorial
                video featuring the <em>GlaucoScan</em> 3D-printed adapter.
              </p>
            </div>
          </div>
        </section>


        {/* Video Tutorial (current placeholder) */}
        
        <section className="bg-[#1a2942] p-8 rounded-xl mb-10">
          <h2 id="tutorial-heading" className="text-2xl font-semibold text-white mb-4">
            Watch the Tutorial
          </h2>
          <div className="rounded-xl overflow-hidden shadow-lg" style={{ position: "relative", paddingBottom: "56.25%" }}>
            <iframe
              src={CURRENT_TUTORIAL_EMBED}
              title="How to Capture a Fundus Image"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              className="w-full h-full"
              style={{ position: "absolute", top: 0, left: 0 }}
            />
          </div>
          <p className="text-sm text-gray-300 mt-3">
            Video courtesy of{" "}
            <a
              href="https://www.youtube.com/@JulianEspinosaOculoplastico"
              target="_blank"
              className="underline text-blue-300"
              rel="noreferrer"
            >
              Julián Espinosa @JulianEspinosaOculoplastico
            </a>
          </p>
        </section>


        {/* Quick tips visible now */}
        {/* Quick Tips (card) */}
        <section className="bg-[#1a2942] p-8 rounded-xl text-gray-300 space-y-5">
          <p>
            To use GlaucoScan.ai, you’ll need a clear retinal image (fundus photo) of one eye.
            You can capture this using a smartphone, a 20D or 28D handheld lens, and our low-cost
            3D-printed adapter. The image should clearly show the optic nerve and surrounding retina.
          </p>
          <p>
            Hold the lens 2–4 cm in front of the patient’s eye, then align your smartphone flashlight
            through the lens until the retina is visible. Adjust angles and distance to find the clearest
            view. Dilation is recommended for optimal clarity.
          </p>
          <p>
            Use our adapter to maintain stable alignment between the light source and the lens for
            consistent imaging. Switch to video mode (with flash), move slowly, pause when the optic disc
            is in focus, then grab a screenshot of the clearest frame to upload.
          </p>
          <p>
            For best results, dim the room and ask the subject to fixate on a still object. Keeping both eyes
            open often reduces blinking and helps stabilization.
          </p>
        </section>

        {/*
          //Step-by-step (enable after your own video is ready)
        <section className="mt-10">
          <button
            type="button"
            onClick={() => setOpenSteps((v) => !v)}
            aria-expanded={openSteps}
            className="w-full text-left bg-[#1a2942] border border-blue-400/30 rounded-xl px-5 py-4 flex items-center justify-between"
          >
            <span className="text-white font-semibold">Step-by-Step Walkthrough (with GlaucoScan Adapter)</span>
            <span className="text-xs px-2 py-1 rounded bg-blue-500/20 border border-blue-400/40 text-blue-100">
              {openSteps ? "Hide" : "Show"}
            </span>
          </button>

          {openSteps && (
            <div className="border border-blue-400/30 border-t-0 rounded-b-xl p-5 bg-[#0f2138]">
              //When your video is ready, you can also add time-stamped anchors here
              <ol className="list-decimal list-inside space-y-2 text-gray-200">
                <li>Attach the GlaucoScan adapter securely to your smartphone.</li>
                <li>Insert the +20D lens (or +28D) into the adapter’s lens holder.</li>
                <li>Position the lens approximately ~8 inches (≈20 cm) from the eye.</li>
                <li>Adjust lighting / coaxial LED for an evenly illuminated view.</li>
                <li>Stabilize, focus on the optic disc, then capture and upload to GlaucoScan.ai.</li>
              </ol>
              <p className="text-sm text-gray-400 mt-3">
                Tip: If reflections appear, slightly change the tilt or distance while keeping the optic disc centered.
              </p>
            </div>
          )}

          //Placeholder banner until your own video replaces the embed
          <div className="mt-4 text-xs text-gray-400">
            Once our demonstration video is published, this section will include
            time-stamped steps synchronized with the video.
          </div>
        </section>
        */}

        {/* Cross-link to hardware (you can add /adapter later) */}
        <div className="mt-10 text-center">
          <Link
            href="/adapter"
            className="inline-block rounded-lg bg-blue-500 hover:bg-blue-600 text-white font-semibold px-6 py-3 transition"
          >
            Download the 3D-Printed Adapter here
          </Link>
        </div>
      </div>
    </div>
  );
}
