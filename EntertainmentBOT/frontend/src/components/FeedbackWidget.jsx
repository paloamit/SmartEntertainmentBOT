import { useState } from 'react'

/**
 * Every recommendation invites feedback:
 *   Satisfied   -> a 5-star rating of the app
 *   Dissatisfied -> an optional review box
 *   ✕            -> dismiss entirely, without answering
 *
 * Nothing is shown once submitted — no "thanks" banner, just the widgets.
 */
function Stars({ onRate }) {
  const [hover, setHover] = useState(0)
  const active = hover || 0
  return (
    <div className="fb-rating">
      <span className="fb-rating-label" id="fb-rate-label">
        How would you rate Smart Entertainment BOT?
      </span>
      <div className="stars">
        {[1, 2, 3, 4, 5].map((n) => (
          <button
            key={n}
            type="button"
            className={`star ${n <= active ? 'on' : ''}`}
            onMouseEnter={() => setHover(n)}
            onMouseLeave={() => setHover(0)}
            onClick={() => onRate(n)}
            title={`${n} star${n > 1 ? 's' : ''}`}
            aria-label={`Rate ${n} out of 5`}
            data-star={n}
          >
            ★
          </button>
        ))}
      </div>
    </div>
  )
}

export default function FeedbackWidget({ onSubmit }) {
  const [stage, setStage] = useState('ask') // ask | rate | review | done | dismissed
  const [comment, setComment] = useState('')
  const [stars, setStars] = useState(0)

  const pick = (rating) => {
    if (rating === 'dissatisfied') setStage('review')
    else setStage('rate')
  }

  const sendRating = (n) => {
    setStars(n)
    onSubmit('satisfied', '', n)
    setStage('done')
  }

  const sendReview = () => {
    onSubmit('dissatisfied', comment.trim())
    setStage('done')
  }

  if (stage === 'dismissed' || stage === 'done') return null

  if (stage === 'rate') {
    return (
      <div className="fb-panel">
        <Stars onRate={sendRating} />
        <button className="fb-close fb-skip-rating" onClick={() => setStage('dismissed')}>
          ✕
        </button>
      </div>
    )
  }

  if (stage === 'review') {
    return (
      <div className="fb-panel">
        <div className="fb-h">
          Sorry that missed the mark — what would’ve been better? (optional)
        </div>
        <textarea
          rows={3}
          value={comment}
          placeholder="e.g. I wanted Korean thrillers from 2023, not documentaries…"
          onChange={(e) => setComment(e.target.value)}
          autoFocus
        />
        <div className="fb-actions">
          <button className="fb-submit" onClick={sendReview}>Submit review</button>
          <button className="fb-cancel" onClick={sendReview}>Skip</button>
          <button className="fb-close" style={{ marginLeft: 'auto' }}
            onClick={() => setStage('dismissed')} title="Dismiss">✕</button>
        </div>
      </div>
    )
  }

  return (
    <div className="feedback">
      <span className="f-label">Was this helpful?</span>
      {/* Dismiss sits on the left, beside the buttons, so it is actually
          reachable instead of floating off at the far edge of a wide screen. */}
      <button className="fb-close" onClick={() => setStage('dismissed')} title="Dismiss">
        ✕
      </button>
      <button className="fb-btn yes" onClick={() => pick('satisfied')}>👍 Satisfied</button>
      <button className="fb-btn no" onClick={() => pick('dissatisfied')}>👎 Dissatisfied</button>
    </div>
  )
}
