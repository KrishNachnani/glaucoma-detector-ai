/** @type {import('next').NextConfig} */
const nextConfig = {
  // Configure the server to handle requests properly
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: { 
    unoptimized: true,
    domains: ['localhost']
  },
  env: {
    NEXT_PUBLIC_GLAUCOMA_API_URL: process.env.GLAUCOMA_PUBLIC_API_URL || process.env.NEXT_PUBLIC_GLAUCOMA_API_URL,
  },
  // Remove static export configuration
  // Add webpack configuration to handle Node.js modules
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // Don't attempt to import server-only modules on the client-side
      config.resolve.fallback = {
        ...config.resolve.fallback,
        net: false,
        tls: false,
        fs: false,
        dns: false,
        child_process: false,
        http2: false,
      };
    }
    return config;
  },
};

module.exports = nextConfig;