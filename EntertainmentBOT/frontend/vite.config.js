import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Only affects local dev; in production the frontend just calls the
    // absolute VITE_API_BASE_URL, so the two deploy independently.
    cors: true,
  },
})
