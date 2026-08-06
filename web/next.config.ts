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

  // the search/intro/stats routes read data/index.json via fs at request
  // time; Next's build-time file tracer usually infers this on its own,
  // but pinning it explicitly guarantees the data file ships with the
  // serverless bundle on Vercel/other providers that trace per-route
  outputFileTracingIncludes: {
    "/api/search": ["./data/index.json"],
    "/api/intro": ["./data/index.json"],
    "/api/stats": ["./data/index.json"],
  },
};

export default nextConfig;
