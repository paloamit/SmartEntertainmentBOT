import { useCallback, useEffect, useRef, useState } from 'react'
import {
  bootstrap, createSession, deleteSession as apiDeleteSession, getSession,
  identifyImage, listSessions, sendFeedback, streamChat,
} from './api.js'
import Sidebar from './components/Sidebar.jsx'
import ChatWindow from './components/ChatWindow.jsx'
import Composer from './components/Composer.jsx'
import Banner from './components/Banner.jsx'
import BootSkeleton from './components/BootSkeleton.jsx'
import SyncDialog from './components/SyncDialog.jsx'

const greetingMsg = (g) => ({
  role: 'assistant',
  kind: 'greeting',
  text: g.greeting,
  titles: [],
  suggestions: g.suggestions || [],
})

export default function App() {
  const [booting, setBooting] = useState(true)
  const [offline, setOffline] = useState('')
  const [sessions, setSessions] = useState([])
  const [sessionId, setSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [limits, setLimits] = useState({ max_sessions: 5, max_queries_per_session: 10 })
  const [queryCount, setQueryCount] = useState(0)
  const [userName, setUserName] = useState(null)
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [visionAvailable, setVisionAvailable] = useState(true)
  const [banner, setBanner] = useState(null)
  const [syncOpen, setSyncOpen] = useState(false)
  const [hasSyncCode, setHasSyncCode] = useState(false)

  const startedAt = useRef(0)

  /* ------------------------------------------------------------------ boot */
  const boot = useCallback(async () => {
    try {
      const d = await bootstrap(true)
      setSessions(d.sessions || [])
      setLimits(d.limits || limits)
      setUserName(d.display_name)
      setHasSyncCode(!!d.has_sync_code)
      setVisionAvailable(d.vision_available !== false)

      // A returning visitor gets their last conversation back, topped with a
      // greeting that uses the name they gave us.
      if (d.active_session) {
        setSessionId(d.active_session.id)
        setQueryCount(d.active_session.query_count || 0)
        setUserName(d.active_session.user_name || d.display_name)
        setMessages([greetingMsg(d.greeting), ...(d.active_session.messages || [])])
      } else {
        setSessionId(null)
        setQueryCount(0)
        setMessages([greetingMsg(d.greeting)])
      }
      setOffline('')
    } catch (e) {
      setOffline(
        "Can't reach the Smart Entertainment BOT server. Check that the backend is running and that VITE_API_BASE_URL is correct.",
      )
      setMessages([])
    } finally {
      setBooting(false)
    }
  }, [])

  const didInit = useRef(false)
  useEffect(() => {
    if (didInit.current) return // avoid a double run under StrictMode
    didInit.current = true
    boot()
  }, [boot])

  const refreshSessions = async () => {
    try {
      setSessions(await listSessions())
    } catch {
      /* non-fatal */
    }
  }

  /* ---------------------------------------------------------- session ops */
  const newChat = async () => {
    if (sessions.length >= limits.max_sessions) {
      setBanner({
        tone: 'warn',
        icon: '🎬',
        text: `You've reached your limit of ${limits.max_sessions} saved chats. Delete one from the panel and I'll start a fresh conversation.`,
      })
      return
    }
    setBanner(null)
    try {
      await createSession()
    } catch (e) {
      if (e.code === 'session_limit') {
        setBanner({ tone: 'warn', icon: '🎬', text: e.message })
        return
      }
    }
    const d = await bootstrap(false)
    setSessions(d.sessions)
    setSessionId(null)
    setQueryCount(0)
    setMessages([greetingMsg(d.greeting)])
  }

  const openSession = async (id) => {
    setBanner(null)
    try {
      const s = await getSession(id)
      setSessionId(s.id)
      setQueryCount(s.query_count || 0)
      setUserName(s.user_name)
      // Keep the personalised greeting on top of a restored conversation.
      const d = await bootstrap(false)
      setMessages([greetingMsg(d.greeting), ...(s.messages || [])])
    } catch {
      /* ignore */
    }
  }

  const removeSession = async (id) => {
    try {
      await apiDeleteSession(id)
    } catch {
      /* ignore */
    }
    const wasActive = id === sessionId
    setBanner(null)
    if (wasActive) {
      const list = await listSessions()
      setSessions(list)
      if (list.length) {
        const s = await getSession(list[0].id)
        const d = await bootstrap(false)
        setSessionId(s.id)
        setQueryCount(s.query_count || 0)
        setMessages([greetingMsg(d.greeting), ...(s.messages || [])])
      } else {
        const d = await bootstrap(false)
        setSessionId(null)
        setQueryCount(0)
        setMessages([greetingMsg(d.greeting)])
      }
    } else {
      refreshSessions()
    }
  }

  /* --------------------------------------------------------------- helpers */
  const updateLast = (fn) =>
    setMessages((m) => {
      const copy = [...m]
      copy[copy.length - 1] = fn(copy[copy.length - 1])
      return copy
    })

  const submitFeedback = async (index, rating, comment = '', appRating = null) => {
    const msg = messages[index]
    if (!msg) return
    let question = ''
    for (let j = index - 1; j >= 0; j--) {
      if (messages[j].role === 'user') { question = messages[j].text; break }
    }
    sendFeedback({
      session_id: sessionId, message_id: msg.id, rating, question,
      answer: msg.text, comment, app_rating: appRating, titles: msg.titles || [],
    })
  }

  const showLimitError = (err) => {
    if (err.code === 'session_limit') {
      setBanner({ tone: 'warn', icon: '🎬', text: err.message })
      refreshSessions()
      return true
    }
    if (err.code === 'query_limit') {
      setBanner({ tone: 'warn', icon: '🎬', text: err.message })
      refreshSessions()
      return true
    }
    if (err.status === 415 || err.code === 'unsupported_type') {
      setBanner({ tone: 'warn', icon: '🖼️', text: err.message })
      return true
    }
    if (err.status === 413) {
      setBanner({ tone: 'warn', icon: '📦', text: err.message })
      return true
    }
    return false
  }

  /* ------------------------------------------------------------- streaming */
  const send = async (text, answers = null) => {
    const msg = (text || '').trim()
    if ((!msg && !answers) || loading || uploading) return
    setBanner(null)

    setMessages((m) => [
      ...m,
      { role: 'user', kind: 'user', text: msg || 'My picks:' },
      {
        role: 'assistant', kind: 'answer', text: '', titles: [], suggestions: [],
        status: '', startedAt: Date.now(),
      },
    ])
    if (answers) {
      setMessages((m) =>
        m.map((x, i) => (i === m.length - 2 ? { ...x, hasAnswers: true } : x)),
      )
    }
    setLoading(true)
    startedAt.current = performance.now()

    try {
      await streamChat({ message: msg || 'Suggest something great.', sessionId, answers }, (ev) => {
        switch (ev.type) {
          case 'meta':
            if (ev.session_id) setSessionId(ev.session_id)
            if (ev.query_count != null) setQueryCount(ev.query_count)
            if (ev.user_name) setUserName(ev.user_name)
            break
          case 'status':
            updateLast((a) => ({ ...a, status: ev.text }))
            break
          case 'token':
            updateLast((a) => ({ ...a, status: '', text: a.text + ev.text }))
            break
          case 'titles':
            updateLast((a) => ({ ...a, status: '', titles: ev.titles || [] }))
            break
          case 'clarify':
            updateLast((a) => ({
              ...a, status: '', kind: 'clarify',
              clarify: ev.clarify, suggestions: ev.suggestions || [],
            }))
            break
          case 'done': {
            if (ev.session_id) setSessionId(ev.session_id)
            if (ev.user_name) setUserName(ev.user_name)
            updateLast((a) => ({
              ...a,
              status: '',
              elapsed: (performance.now() - startedAt.current) / 1000,
              cached: !!ev.cached,
              kind: a.clarify ? 'clarify' : 'answer',
              // Every suggestion / recommendation invites feedback.
              askFeedback: ev.kind === 'answer',
            }))
            refreshSessions()
            break
          }
          default:
            break
        }
      })
    } catch (err) {
      if (!showLimitError(err)) {
        updateLast((a) => ({
          ...a, status: '',
          text: a.text || '⚠️ I couldn’t reach the server. Please try again.',
        }))
      }
    } finally {
      setLoading(false)
    }
  }

  /** Answer the clarifying widget without typing anything. */
  const submitClarify = (index, payload) => {
    setMessages((m) =>
      m.map((x, i) => (i === index ? { ...x, clarifyAnswered: true } : x)),
    )
    const parts = Object.entries(payload.answers || {})
      .filter(([k]) => !k.startsWith('_'))
      .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${Array.isArray(v) ? v.join(', ') : v}`)
    send(parts.length ? `Based on my picks — ${parts.join('; ')}` : 'Surprise me', payload.answers)
  }

  /* -------------------------------------------------------- image upload */
  const upload = async (file, hint = '') => {
    if (uploading) return
    setUploading(true)
    setBanner(null)
    // Show what the user actually typed. Their prompt was being silently
    // dropped whenever an image was attached.
    const said = hint.trim()
    const label = said
      ? `${said}\n\n📷 ${file.name}`
      : `📷 Identify this poster: ${file.name}`
    setMessages((m) => [
      ...m,
      { role: 'user', kind: 'user', text: label },
      {
        role: 'assistant', kind: 'answer', text: '', titles: [], suggestions: [],
        status: 'Looking at the artwork…', startedAt: Date.now(),
      },
    ])
    const t0 = performance.now()

    try {
      const r = await identifyImage(file, sessionId, said)
      if (!r.ok) {
        setMessages((m) => {
          const copy = [...m]
          copy[copy.length - 1] = {
            ...copy[copy.length - 1],
            status: '',
            text: r.message || 'I couldn’t identify that one.',
            elapsed: (performance.now() - t0) / 1000,
          }
          return copy
        })
        return
      }
      const titles = [r.title, ...(r.alternatives || [])].filter(Boolean)
      setMessages((m) => {
        const copy = [...m]
        copy[copy.length - 1] = {
          ...copy[copy.length - 1],
          status: '',
          text: r.text || `That's **${r.title.title}**.`,
          titles,
          elapsed: (performance.now() - t0) / 1000,
          askFeedback: true,
        }
        return copy
      })
      refreshSessions()
    } catch (err) {
      if (!showLimitError(err)) {
        setMessages((m) => {
          const copy = [...m]
          copy[copy.length - 1] = {
            ...copy[copy.length - 1],
            status: '',
            text: err.message || '⚠️ That upload didn’t work. Please try again.',
            elapsed: (performance.now() - t0) / 1000,
          }
          return copy
        })
      }
    } finally {
      setUploading(false)
    }
  }

  /* ----------------------------------------------------------------- view */
  const quotaLeft = Math.max(0, limits.max_queries_per_session - queryCount)
  const quotaTone = quotaLeft === 0 ? 'crit' : quotaLeft <= 3 ? 'warn' : ''

  return (
    <div className="app">
      <main className="chat-pane">
        <header className="topbar">
          <span className="logo">🎬 Smart Entertainment BOT</span>
          <span className="tag">Movies · Web Series · Animation</span>
          <span className="topbar-spacer" />
          <span className={`quota ${quotaTone}`} title="Questions used in this chat">
            <span className="dot" />
            <b>{queryCount}</b>/{limits.max_queries_per_session} questions
          </span>
          <button className="icon-btn" onClick={() => setSyncOpen(true)} title="Sync across devices">
            🔗
          </button>
        </header>

        {offline && <div className="offline">⚠️ {offline}</div>}

        {banner && (
          <Banner
            tone={banner.tone}
            icon={banner.icon}
            text={banner.text}
            onClose={() => setBanner(null)}
          />
        )}

        {booting ? (
          <BootSkeleton />
        ) : (
          <ChatWindow
            messages={messages}
            loading={loading || uploading}
            onSuggestion={(s) => send(s)}
            onFeedback={submitFeedback}
            onClarifySubmit={submitClarify}
          />
        )}

        <Composer
          onSend={send}
          onUpload={upload}
          disabled={loading || uploading}
          uploading={uploading}
          visionAvailable={visionAvailable}
          quotaLeft={quotaLeft}
        />
      </main>

      <Sidebar
        sessions={sessions}
        activeId={sessionId}
        limits={limits}
        onSelect={openSession}
        onNew={newChat}
        onDelete={removeSession}
        onSync={() => setSyncOpen(true)}
        syncing={hasSyncCode}
      />

      {syncOpen && (
        <SyncDialog
          hasCode={hasSyncCode}
          onClose={() => setSyncOpen(false)}
          onSynced={async (d) => {
            setSyncOpen(false)
            setHasSyncCode(true)
            setSessions(d.sessions || [])
            await boot()
          }}
        />
      )}
    </div>
  )
}
