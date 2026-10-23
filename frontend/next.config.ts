import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The dev server is opened as 127.0.0.1 (the only address compose publishes); without this it blocks its own dev scripts.
  allowedDevOrigins: ["127.0.0.1"],
  // The browser calls /api/* on this origin and Next forwards it to the API container, so no CORS.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${process.env.API_URL}/:path*` }];
  },
};

export default nextConfig;
