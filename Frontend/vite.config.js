
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',   // use loopback to avoid hostname oddities
    port: 5173,
    strictPort: true,     // fail fast if 5173 is busy
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/reindex': 'http://127.0.0.1:8000',
      '/search': 'http://127.0.0.1:8000',
      '/sessions': 'http://127.0.0.1:8000',
      '/chat': 'http://127.0.0.1:8000',
      '/chat-file': 'http://127.0.0.1:8000'
    }
  }
});
