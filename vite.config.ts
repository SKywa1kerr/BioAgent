import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(async () => ({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: "127.0.0.1",
    hmr: {
      host: "127.0.0.1",
    },
    watch: {
      ignored: ["**/electron/**"],
    },
  },
}));
