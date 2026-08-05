import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // allow the demo to be shared through ngrok tunnels in dev mode —
  // without this, remote browsers get the HTML but the JS bundles are
  // blocked cross-origin, so the page never hydrates and the search
  // form falls back to a native submit (the "page just reloads" bug)
  allowedDevOrigins: [
    "asymmetric-swaggeringly-drema.ngrok-free.dev",
    "*.ngrok-free.dev",
    "*.ngrok.io",
  ],
};

export default nextConfig;
