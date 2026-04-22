import path from "node:path";
import type { NextConfig } from "next";

/**
 * The Flask backend at ../webapp/app.py listens on 127.0.0.1:5000 and exposes:
 *   POST /api/generate
 *   GET  /api/list
 *   GET  /generated/<run_id>/output.csd
 *   GET  /generated/<run_id>/output.wav
 *
 * We proxy both namespaces in dev so the browser stays same-origin and the
 * <audio> element can read the WAV without CORS gymnastics.
 */
const BACKEND =
  process.env.CSTORE_BACKEND_URL ?? "http://127.0.0.1:5000";

const config: NextConfig = {
  // Pin Turbopack to this package — avoids ambiguity when an unrelated
  // package-lock.json exists higher up the file system.
  turbopack: {
    root: path.resolve(import.meta.dirname ?? "."),
  },
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${BACKEND}/api/:path*` },
      { source: "/generated/:path*", destination: `${BACKEND}/generated/:path*` },
    ];
  },
};

export default config;
