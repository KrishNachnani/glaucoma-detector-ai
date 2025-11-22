'use client';

import { useState } from 'react';

const WEB3FORMS_ACCESS_KEY = process.env.NEXT_PUBLIC_WEB3FORMS_ACCESS_KEY;

export default function ContactPage() {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setSuccessMessage('');
    setErrorMessage('');

    if (!WEB3FORMS_ACCESS_KEY) {
      setErrorMessage('Contact form is not configured. Please set NEXT_PUBLIC_WEB3FORMS_ACCESS_KEY.');
      return;
    }

    setIsSubmitting(true);

    try {
      const formData = new FormData(e.currentTarget);
      formData.append('access_key', WEB3FORMS_ACCESS_KEY);

      const response = await fetch('https://api.web3forms.com/submit', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (result.success) {
        setSuccessMessage('Thank you! Your message has been sent.');
        (e.target as HTMLFormElement).reset();
      } else {
        setErrorMessage('Failed to send message. Please try again.');
      }
    } catch (err) {
      console.error('Web3Forms error:', err);
      setErrorMessage('An unexpected error occurred. Please try again later.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-[#0a192f] to-[#112240] py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center text-white mb-2">Contact Us</h1>
        <p className="text-center text-blue-300 mb-10">
          We&apos;d love to hear from you, whether it&apos;s feedback, collaboration, or questions.
        </p>

        <section className="bg-[#0b1930] rounded-2xl p-8 shadow-lg">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="name" className="block text-sm font-medium text-blue-200 mb-1">
                Name
              </label>
              <input
                id="name"
                name="name"
                type="text"
                required
                className="w-full rounded-md bg-[#0f213a] border border-[#1f3355] text-white px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label htmlFor="email" className="block text-sm font-medium text-blue-200 mb-1">
                Email
              </label>
              <input
                id="email"
                name="email"
                type="email"
                required
                className="w-full rounded-md bg-[#0f213a] border border-[#1f3355] text-white px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label htmlFor="message" className="block text-sm font-medium text-blue-200 mb-1">
                Message
              </label>
              <textarea
                id="message"
                name="message"
                rows={5}
                required
                className="w-full rounded-md bg-[#0f213a] border border-[#1f3355] text-white px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full flex items-center justify-center rounded-md bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2.5 transition disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {isSubmitting ? 'Sending...' : 'Send Message'}
            </button>

            {successMessage && (
              <p className="text-sm text-green-400 text-center mt-2">{successMessage}</p>
            )}
            {errorMessage && (
              <p className="text-sm text-red-400 text-center mt-2">{errorMessage}</p>
            )}
          </form>
        </section>
      </div>
    </main>
  );
}
