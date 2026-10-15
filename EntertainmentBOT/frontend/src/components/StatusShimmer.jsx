/** Shimmering pill shown while the bot is thinking or calling a tool. */
export default function StatusShimmer({ text }) {
  if (!text) return null
  return (
    <div className="typing-status shimmer">
      <span className="shimmer-bar" />
      <span className="shimmer-text">{text}</span>
    </div>
  )
}

/** Stages we cycle through so the UI always feels alive during a long turn. */
export const THINKING_STAGES = [
  'Thinking about it…',
  'Searching the catalogs…',
  'Weighing up the options…',
  'Almost done…',
]

/**
 * Picks what to show while waiting: the backend's own status when it has sent
 * one, otherwise a staged fallback that advances with elapsed time.
 */
export function pickStatus(serverStatus, elapsedMs, gotTokens) {
  if (serverStatus) return serverStatus
  if (gotTokens) return 'Writing your reply…'
  const idx = Math.min(Math.floor(elapsedMs / 1600), THINKING_STAGES.length - 1)
  return THINKING_STAGES[idx]
}
