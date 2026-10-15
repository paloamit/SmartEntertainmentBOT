import { useCallback, useRef, useState } from 'react'

/**
 * The composer: a "+" button for poster uploads (images only — video is
 * rejected before it ever leaves the browser) plus the message box.
 *
 * Images can also be dropped straight onto the composer, which is what people
 * naturally try first — dragging a poster out of a browser tab and dropping
 * it into the chat is a muscle-memory move from every other chat app.
 */
export default function Composer({ onSend, onUpload, disabled, quotaLeft, uploading, visionAvailable = true }) {
  const [text, setText] = useState('')
  const [file, setFile] = useState(null)
  const [err, setErr] = useState('')
  const [dragging, setDragging] = useState(false)
  const taRef = useRef(null)
  const fileRef = useRef(null)
  // dragenter/dragleave fire for every child element, so nesting counts
  // are tracked to avoid the highlight flickering as the pointer crosses
  // over the textarea and buttons.
  const dragDepth = useRef(0)

  const grow = (el) => {
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 168)}px`
  }

  const accept = useCallback((f) => {
    if (!f) return
    if (!f.type.startsWith('image/')) {
      setErr('Only images are supported — please drop a JPG, PNG or WEBP.')
      setFile(null)
      return
    }
    setErr('')
    setFile(f)
  }, [])

  const pick = (e) => {
    const f = e.target.files?.[0]
    e.target.value = '' // allow re-picking the same file
    accept(f)
  }

  const clearFile = () => {
    setFile(null)
    setErr('')
  }

  /* ------------------------------------------------------------ drag-drop */
  // How an image reaches us depends entirely on where it came from:
  //
  //   Finder/Desktop  -> types ['Files'],        dataTransfer.files is set
  //   Browser tab     -> types ['text/uri-list'], only a REMOTE url
  //   Another app     -> often neither
  //
  // Checking only for 'Files' means dragging a poster straight out of a
  // browser tab — the most natural thing to try — was silently ignored.
  const dragCarriesImage = (dt) => {
    const types = dt?.types
    if (!types) return false
    const list = Array.from(types)
    return list.includes('Files') ||
      list.includes('text/uri-list') ||
      list.includes('text/html') ||
      list.includes('text/plain')
  }

  const looksLikeImageUrl = (url) =>
    /\.(jpe?g|png|webp|gif|avif|bmp)(\?|#|$)/i.test(url || '') ||
    /(image|img|photo|cdn|media)/i.test(url || '')

  // Pull a remote image down so it can go through the same upload path as a
  // local file. CORS can block this for some hosts; the user then gets a
  // clear message instead of a silent no-op.
  const remoteImageToFile = async (url) => {
    const resp = await fetch(url, { mode: 'cors' })
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const blob = await resp.blob()
    if (!blob.type.startsWith('image/')) {
      throw new Error(`not an image (${blob.type || 'unknown type'})`)
    }
    const name = (url.split('/').pop() || 'dropped-image').split('?')[0] || 'dropped-image'
    return new File([blob], name, { type: blob.type })
  }

  const onDragEnter = (e) => {
    if (!dragCarriesImage(e.dataTransfer)) return
    e.preventDefault()
    dragDepth.current += 1
    setDragging(true)
  }

  const onDragOver = (e) => {
    if (!dragCarriesImage(e.dataTransfer)) return
    // Without preventDefault the browser just navigates to the dropped URL.
    e.preventDefault()
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy'
  }

  const onDragLeave = (e) => {
    e.preventDefault()
    dragDepth.current = Math.max(0, dragDepth.current - 1)
    if (dragDepth.current === 0) setDragging(false)
  }

  const onDrop = async (e) => {
    e.preventDefault()
    dragDepth.current = 0
    setDragging(false)
    if (!visionAvailable) {
      setErr("Poster recognition isn't configured here — type the title instead.")
      return
    }

    const dt = e.dataTransfer
    const local = dt?.files?.[0]
    if (local) {
      accept(local)
      return
    }

    // Dragged out of a browser tab or another app: only a URL comes with it.
    const raw = dt?.getData('text/uri-list') || dt?.getData('text/plain') || ''
    const url = raw.split('\n').map((s) => s.trim()).filter(Boolean)[0]
    if (!url || !/^https?:/i.test(url)) {
      setErr("I couldn't read that drop — try an image file, or a link ending in .jpg/.png.")
      return
    }
    try {
      accept(await remoteImageToFile(url))
    } catch (err) {
      setErr(
        `I couldn't download that image${looksLikeImageUrl(url) ? '' : ''} ` +
        `(${err.message}). Save it and drop the file instead.`,
      )
    }
  }

  // Copy-paste an image straight into the prompt.
  const onPaste = (e) => {
    if (!visionAvailable) return
    const item = [...(e.clipboardData?.items || [])]
      .find((i) => i.type.startsWith('image/'))
    if (!item) return
    const f = item.getAsFile()
    if (f) {
      e.preventDefault()
      accept(f)
    }
  }

  const submit = () => {
    const msg = text.trim()
    if (!msg && !file) return

    if (file) {
      // Hand the typed text along as a hint rather than silently dropping it.
      onUpload(file, msg)
      clearFile()
      setText('')
      if (taRef.current) taRef.current.style.height = 'auto'
      return
    }
    onSend(msg)
    setText('')
    if (taRef.current) taRef.current.style.height = 'auto'
  }

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const blocked = disabled || uploading || quotaLeft === 0

  return (
    <div className="composer">
      {(file || err) && (
        <div className="upload-note" style={err ? { color: '#ffdc9e', background: 'rgba(245,165,36,.12)', borderColor: 'rgba(245,165,36,.35)' } : undefined}>
          <span>{err ? '⚠️' : '🖼️'}</span>
          <span style={{ flex: 1 }}>
            {err || `Ready to identify: ${file.name}`}
          </span>
          {!err && (
            <button onClick={clearFile} title="Remove image"
              style={{ color: 'inherit', padding: '0 4px', fontSize: 14 }}>✕</button>
          )}
        </div>
      )}

      <div
        className={`composer-inner ${dragging ? 'drop-target' : ''}`}
        onDragEnter={onDragEnter}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        <button
          className="attach"
          title={
            visionAvailable
              ? 'Upload a poster or still (images only)'
              : 'Poster recognition isn’t configured on this setup — type the title instead'
          }
          onClick={() => fileRef.current?.click()}
          disabled={disabled || uploading || !visionAvailable}
        >
          +
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          onChange={pick}
          style={{ display: 'none' }}
        />

        <textarea
          ref={taRef}
          rows={1}
          value={text}
          disabled={disabled}
          placeholder={
            quotaLeft === 0
              ? 'Query limit reached — start a new chat to continue 🎬'
              : 'Ask for a movie, series or anime…'
          }
          onChange={(e) => { setText(e.target.value); grow(e.target) }}
          onKeyDown={onKey}
          onPaste={onPaste}
        />

        <button className="send" onClick={submit} disabled={blocked}
          title={uploading ? 'Identifying…' : 'Send'}>
          {uploading ? <span className="mini-spin" /> : '➤'}
        </button>
      {dragging && (
          <div className="drop-hint">
            <span aria-hidden>🖼️</span>
            <span>Drop the poster to identify it</span>
          </div>
        )}
      </div>

      <div className="composer-note">
        <b>Enter</b> to send · <b>Shift+Enter</b> for a new line ·{' '}
        <b>+</b> to identify a poster (images only){visionAvailable ? ' — or drag, drop & paste' : ' — unavailable'}
      </div>
    </div>
  )
}
