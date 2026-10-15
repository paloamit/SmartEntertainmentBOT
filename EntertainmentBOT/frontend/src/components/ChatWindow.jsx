import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import TitleCard from './TitleCard.jsx'
import ClarifyCard from './ClarifyCard.jsx'
import FeedbackWidget from './FeedbackWidget.jsx'
import StatusShimmer, { pickStatus } from './StatusShimmer.jsx'
import { getTitle } from '../api.js'

export default function ChatWindow({ messages, loading, onSuggestion, onFeedback, onClarifySubmit }) {
  const endRef = useRef(null)
  // A single `open` object guarantees only ONE card is expanded in the whole chat.
  const [open, setOpen] = useState(null)
  const [now, setNow] = useState(Date.now())

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Drives the shimmer text while we wait for the first token.
  useEffect(() => {
    if (!loading) return
    const t = setInterval(() => setNow(Date.now()), 500)
    return () => clearInterval(t)
  }, [loading])

  const openSimilar = async (msgIndex, key) => {
    try {
      const card = await getTitle(key)
      if (card) setOpen({ i: msgIndex, key, card })
    } catch {
      /* non-fatal */
    }
  }

  const isOnlyGreeting = messages.length === 1 && messages[0].kind === 'greeting'

  return (
    <div className="messages">
      {isOnlyGreeting && (
        <div className="hero">
          <div className="h-emoji">🎬</div>
          <h2>What are we watching today?</h2>
          <p>Movies, web series or anime — I'll find your next favourite.</p>
        </div>
      )}

      {messages.map((m, i) => {
        const isLast = i === messages.length - 1
        const streaming = m.role === 'assistant' && loading && isLast
        const showDots = streaming && !m.text
        const showBubble = m.role !== 'assistant' || Boolean(m.text) || showDots
        // Once the picks are on screen the answer is visibly complete. The
        // model often spends several more seconds in a second tool round, and
        // a "Writing your reply…" pill over finished cards just reads as stuck.
        const showStatus = streaming && !(m.titles || []).length

        // When a card is open, hoist it to the front so it reads as one unit.
        const isOpenMsg = open?.i === i
        const cards = isOpenMsg
          ? [open.card, ...(m.titles || []).filter((t) => t.key !== open.key)]
          : m.titles || []

        return (
          <div key={i} className={`msg ${m.role} ${m.kind || ''}`}>
            {showBubble && (
              <div className="bubble">
                {m.role !== 'assistant' ? (
                  <span>{m.text}</span>
                ) : showDots ? (
                  <span className="typing"><span /><span /><span /></span>
                ) : (
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text || ''}</ReactMarkdown>
                )}
              </div>
            )}

            {showStatus && <StatusShimmer text={pickStatus(m.status, now - (m.startedAt || now), Boolean(m.text))} />}

            {m.role === 'assistant' && m.clarify && !m.clarifyAnswered && (
              <ClarifyCard
                clarify={m.clarify}
                onSubmit={(payload) => onClarifySubmit(i, payload)}
              />
            )}
            {m.role === 'assistant' && m.clarify && m.clarifyAnswered && (
              <div className="cf-answered">
                <span>🎬</span>
                <span>
                  Based on your inputs, here are my recommended picks — best match first.
                </span>
              </div>
            )}

            {/* Cards follow the written reply, so the timing line sits last. */}
            {cards.length > 0 && (
              <div className="titles">
                {cards.map((t) => (
                  <TitleCard
                    key={t.key}
                    title={t}
                    expanded={isOpenMsg && t.key === open.key}
                    onToggle={() =>
                      setOpen((cur) =>
                        cur?.i === i && cur?.key === t.key ? null : { i, key: t.key, card: t },
                      )
                    }
                    onOpenSimilar={(key) => openSimilar(i, key)}
                  />
                ))}
              </div>
            )}

            {/* Timing comes before the feedback prompt. */}
            {m.role === 'assistant' && m.elapsed != null && (
              <div className="msg-meta">
                <span>⏱ {m.elapsed.toFixed(1)}s</span>
                {m.cached && (
                  <span className="badge-cache">⚡ answered from cache — no LLM call</span>
                )}
              </div>
            )}

            {m.role === 'assistant' && m.askFeedback && (
              <FeedbackWidget onSubmit={(rating, comment, appRating) =>
                onFeedback(i, rating, comment, appRating)} />
            )}

            {/* Quick replies are already rendered inside the clarify widget, so
                they must not also appear here as suggestion chips. */}
            {m.role === 'assistant' && m.suggestions?.length > 0 && !m.clarify && (
              <div className="chips">
                {m.suggestions.map((s) => (
                  <button key={s} className="chip" onClick={() => onSuggestion(s)} disabled={loading}>
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        )
      })}
      <div ref={endRef} />
    </div>
  )
}
