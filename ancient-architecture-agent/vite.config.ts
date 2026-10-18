import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig, loadEnv} from 'vite';

export default defineConfig(({mode}) => {
  const env = loadEnv(mode, '.', '');
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modifyâfile watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      host: '127.0.0.1',
      allowedHosts: env.VITE_ALLOWED_HOSTS?.split(',').map(host => host.trim()).filter(Boolean) ?? [],
      port: 5173,
      proxy: {
        '/agent': 'http://127.0.0.1:8001',
        '/models_glb': 'http://127.0.0.1:8000',
        '/models': 'http://127.0.0.1:8000',
        '/outputs': 'http://127.0.0.1:8000',
        '/predict': 'http://127.0.0.1:8000',
        '/health': 'http://127.0.0.1:8000',
      },
    },
  };
});
