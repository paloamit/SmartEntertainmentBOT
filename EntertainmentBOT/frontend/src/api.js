/**
 * The entire backend contract lives here. The React app knows nothing else
 * about the backend, so you can point VITE_API_BASE_URL at any deployment — a
 * local server, a Render URL, or something another team hosts — with no other
 * change.
 */
const BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

/* ------------------------------------------------------------------ identity */
const VISITOR_KEY = 'sebot.visitorId'

/** Stable per-browser id so the backend can find this user's sessions again. */
export function getVisitorId() {
  let id = localStorage.getItem(VISITOR_KEY)
  if (!id) {
    id = (crypto.randomUUID?.() ||
      'v-' + Math.random().toString(36).slice(2) + Date.now().toString(36))
    localStorage.setItem(VISITOR_KEY, id)
  }
  return id
}

export function setVisitorId(id) {
  if (id) localStorage.setItem(VISITOR_KEY, id)
}

function headers(extra = {}) {
  return { 'X-Visitor-Id': getVisitorId(), ...extra }
}

/** Turns a non-2xx response into a readable Error carrying the backend's code. */
async function fail(res) {
  let body = {}
  try {
    body = await res.json()
  } catch {
    /* non-JSON error page */
  }
  const detail = body.detail
  const info = typeof detail === 'object' && detail !== null ? detail : { message: detail }
  const err = new Error(info.message || `Request failed (${res.status})`)
  err.code = info.code
  err.status = res.status
  err.info = info
  return err
}

async function json(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, { ...options, headers: headers(options.headers) })
  if (!res.ok) throw await fail(res)
  return res.json()
}

const post = (path, body) =>
  json(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  })

/* -------------------------------------------------------------------- calls */
export const bootstrap = (resume = true) =>
  json(`/api/bootstrap?resume=${resume}`)

export const createSession = () => post('/api/sessions')

export const listSessions = () => json('/api/sessions')

export const getSession = (id) => json(`/api/sessions/${id}`)

export const deleteSession = (id) => json(`/api/sessions/${id}`, { method: 'DELETE' })

export const sendFeedback = (payload) =>
  post('/api/feedback', payload).catch(() => {}) // best-effort, never block the UI

export const getTitle = (key) => json(`/api/titles/${encodeURIComponent(key)}`)

export const trending = (type = 'all') =>
  json(`/api/browse/trending?media_type=${type}&limit=12`).catch(() => [])

export const saveSyncCode = (passphrase) => post('/api/identity/unlock', { passphrase })

export const clearSyncCode = () => json('/api/identity/unlock', { method: 'DELETE' })

export const restoreFromSyncCode = (passphrase) =>
  post('/api/identity/unlock/restore', { passphrase }).then((d) => {
    setVisitorId(d.visitor_id)
    return d
  })

/** Poster recognition — images only; the backend rejects anything else. */
export async function identifyImage(file, sessionId, hint) {
  const form = new FormData()
  form.append('file', file)
  if (sessionId) form.append('session_id', sessionId)
  // Anything the user typed alongside the image, used to disambiguate
  // between close matches.
  if (hint) form.append('hint', hint)
  const res = await fetch(`${BASE}/api/vision/identify`, {
    method: 'POST',
    headers: headers(), // multipart: do NOT set Content-Type, the browser adds the boundary
    body: form,
  })
  if (!res.ok) throw await fail(res)
  return res.json()
}

/* ------------------------------------------------------------------ streaming */
/**
 * Streams one assistant turn over SSE. Calls onEvent for each frame:
 *   meta | status | token | titles | clarify | done
 */
export function streamChat({ message, sessionId, answers }, onEvent) {
  return fetch(`${BASE}/api/chat/stream`, {
    method: 'POST',
    headers: headers({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ message, session_id: sessionId || null, answers: answers || null }),
  }).then(async (res) => {
    if (!res.ok) throw await fail(res)
    if (!res.body) throw new Error('stream request failed')

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      let idx
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const chunk = buf.slice(0, idx)
        buf = buf.slice(idx + 2)
        const line = chunk.replace(/^data:\s?/, '').trim()
        if (!line) continue
        try {
          onEvent(JSON.parse(line))
        } catch {
          /* ignore a malformed frame rather than breaking the stream */
        }
      }
    }
  })
}

export const API_BASE = BASE
