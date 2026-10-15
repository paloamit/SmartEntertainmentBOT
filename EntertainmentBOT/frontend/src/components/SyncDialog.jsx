import { useState } from 'react'
import { restoreFromSyncCode, saveSyncCode } from '../api.js'

/**
 * Optional "sync code": lets the same person reopen their chats on another
 * device or after clearing browser storage.
 */
export default function SyncDialog({ onClose, hasCode, onSynced }) {
  const [mode, setMode] = useState(hasCode ? 'info' : 'create')
  const [phrase, setPhrase] = useState('')
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async () => {
    if (phrase.trim().length < 4) {
      setErr('Use at least 4 characters.')
      return
    }
    setBusy(true)
    setErr('')
    try {
      if (mode === 'restore') {
        const d = await restoreFromSyncCode(phrase.trim())
        onSynced(d)
      } else {
        await saveSyncCode(phrase.trim())
        setMode('info')
      }
    } catch (e) {
      setErr(e.message || 'Something went wrong.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>🔗 Sync across devices</h3>

        {mode === 'info' ? (
          <>
            <p>
              Your chats are tied to this browser. Enter your sync code on another
              device to pick up right where you left off.
            </p>
            <button
              className="m-primary"
              style={{ width: '100%', marginBottom: 9 }}
              onClick={() => setMode('restore')}
            >
              I have a sync code
            </button>
            <button className="m-ghost" style={{ width: '100%' }} onClick={onClose}>
              Close
            </button>
          </>
        ) : (
          <>
            <p>
              {mode === 'create'
                ? 'Create a sync code to keep your chats safe and reopen them on another device.'
                : 'Enter the sync code you created earlier.'}
            </p>
            <input
              autoFocus
              value={phrase}
              placeholder="e.g. my-movies-2026"
              onChange={(e) => setPhrase(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') { e.preventDefault(); submit() }
              }}
            />
            {err && <div className="err">{err}</div>}
            <div className="m-actions">
              <button className="m-ghost" onClick={onClose}>Cancel</button>
              <button className="m-primary" onClick={submit} disabled={busy}>
                {busy ? 'Working…' : mode === 'create' ? 'Create' : 'Restore'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
