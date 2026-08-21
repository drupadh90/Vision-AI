import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The browser is never on the same host as the backend in a sandboxed preview,
// so the frontend always calls *relative* URLs and Vite proxies them onward.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    // Accept the proxied preview hostname (*.e2b.app) as well as localhost.
    allowedHosts: true,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/ws": { target: "ws://127.0.0.1:8000", ws: true, changeOrigin: true },
    },
  },
});
