import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom"],
          blueprint: ["@blueprintjs/core", "@blueprintjs/select"],
          plotly: ["react-plotly.js", "plotly.js-dist-min"],
        },
      },
    },
  },
});
