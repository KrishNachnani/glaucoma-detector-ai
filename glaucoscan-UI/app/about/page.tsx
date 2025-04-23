export default function About() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-[#0a192f] to-[#112240] py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-white mb-8">About Us</h1>
        
        <div className="space-y-12">
          <section className="bg-[#1a2942] p-8 rounded-xl">
            <h2 className="text-2xl font-semibold text-white mb-4">Our Mission</h2>
            <p className="text-gray-300">
              We are dedicated to revolutionizing eye care through artificial intelligence. Our mission is to make early glaucoma detection accessible to everyone, everywhere, helping prevent vision loss through timely intervention.
            </p>
          </section>
          
          <section className="bg-[#1a2942] p-8 rounded-xl">
            <h2 className="text-2xl font-semibold text-white mb-4">Innovation in Healthcare</h2>
            <p className="text-gray-300">
              By combining cutting-edge AI technology with medical expertise, we've developed a powerful tool that assists healthcare professionals in detecting glaucoma with high accuracy. Our system processes retinal images in seconds, providing quick and reliable results.
            </p>
          </section>
          
          <section className="bg-[#1a2942] p-8 rounded-xl">
            <h2 className="text-2xl font-semibold text-white mb-4">Our Commitment</h2>
            <p className="text-gray-300">
              We're committed to continuous improvement and innovation in eye care technology. Our team of experts works tirelessly to enhance our AI algorithms, ensuring the highest standards of accuracy and reliability in glaucoma detection.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}