import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
  },
  build: {
    rollupOptions: {
      // Externalise Plotly: loaded from CDN in index.html, not bundled.
      external: ["plotly.js-dist-min"],
      output: {
        globals: {
          "plotly.js-dist-min": "Plotly",
        },
        manualChunks: {
          react: ["react", "react-dom"],
          blueprint: ["@blueprintjs/core", "@blueprintjs/select"],
        },
      },
    },
  },
});
