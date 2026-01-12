import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Backend API URL - use environment variable for Docker, default to localhost for local dev
const apiHost = process.env.VITE_API_HOST || '127.0.0.1'
const apiPort = process.env.VITE_API_PORT || '8000'

export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173,
        host: '0.0.0.0',
        proxy: {
            '/api/ws': {
                target: `ws://${apiHost}:${apiPort}`,
                ws: true,
                changeOrigin: true,
            },
            '/api': {
                target: `http://${apiHost}:${apiPort}`,
                changeOrigin: true,
            },
        },
    },
})
