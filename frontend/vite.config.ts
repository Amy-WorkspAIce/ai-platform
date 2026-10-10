import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, proxy API calls to the backend (uvicorn on :8484).
const backend = process.env.AMY_BACKEND_URL ?? "http://localhost:8484";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/v1": backend,
      "/api": backend,
      "/health": backend,
    },
  },
});
