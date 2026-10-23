import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The dev server is opened as 127.0.0.1 (the only address compose publishes); without this it blocks its own dev scripts.
  allowedDevOrigins: ["127.0.0.1"],
};

export default nextConfig;
