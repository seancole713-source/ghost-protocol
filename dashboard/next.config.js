/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    BACKEND_URL: process.env.BACKEND_URL || 'https://ghost-protocol-production.up.railway.app',
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://ghost-protocol-production.up.railway.app/api/:path*',
      },
    ]
  },
}

module.exports = nextConfig
