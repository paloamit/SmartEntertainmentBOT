// Frontend configuration. The app talks to the backend only over HTTP via
// VITE_API_BASE_URL, so the two deploy independently.
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
