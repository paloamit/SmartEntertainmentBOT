import { useEffect, useRef, useState } from 'react'
import { getTitle } from '../api.js'
import WatchLinks from './WatchLinks.jsx'

/* Detail is fetched ONCE per title and reused for the whole app session —
   this is why the LLM never has to ship detail payloads into the chat. */
const detailCache = new Map()
const inFlight = new Map()

function loadDetail(key) {
  if (detailCache.has(key)) return Promise.resolve(detailCache.get(key))
  if (inFlight.has(key)) return inFlight.get(key)
  const p = getTitle(key)
    .then((d) => {
      detailCache.set(key, d)
      inFlight.delete(key)
      return d
    })
    .catch(() => {
      inFlight.delete(key)
      return null
    })
  inFlight.set(key, p)
  return p
}

export function preloadTitle(key) {
  loadDetail(key)
}

/* --------------------------------------------------------------- rating ring */

/* Solid green disc with the score in it. Green is the app's "good" accent, so
   the badge reads as a quality stamp rather than a precise meter. */
function Rating({ score, source }) {
  if (!score) return null
  return (
    <div
      className="rating"
      title={`${score}/10 ${source === 'imdb' ? 'IMDb' : 'user'} rating`}
    >
      <span className="val">{score.toFixed(1)}</span>
    </div>
  )
}

/* ------------------------------------------------------------------ accordion */
function Accordion({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className={`acc-item ${open ? 'open' : ''}`}>
      <button className="acc-head" onClick={() => setOpen((o) => !o)}>
        {title}
        <span className="car">▼</span>
      </button>
      {open && <div className="acc-panel">{children}</div>}
    </div>
  )
}

const typeLabel = (t, seasons) => {
  if (t === 'tv') return seasons ? `Series · S${seasons}` : 'Series'
  if (t === 'animation') return 'Animation'
  return 'Movie'
}

const mins = (n) => (n ? `${n} min` : null)

export default function TitleCard({ title, expanded, onToggle, onOpenSimilar }) {
  const [detail, setDetail] = useState(() => detailCache.get(title.key) || null)
  const [loading, setLoading] = useState(false)
  const [failed, setFailed] = useState(false)
  const ref = useRef(null)

  // Lazy-load the full record the first time the card is opened.
  useEffect(() => {
    if (!expanded || detail || loading) return
    setLoading(true)
    loadDetail(title.key).then((d) => {
      setDetail(d)
      setLoading(!d)
      setFailed(!d)
    })
  }, [expanded, detail, loading, title.key])

  // Scroll a newly opened card into view so long lists stay navigable.
  useEffect(() => {
    if (expanded && ref.current) {
      ref.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [expanded])

  const d = detail || title
  const score = title.imdb_rating || title.rating
  const scoreSrc = title.imdb_rating ? 'IMDb' : 'TMDB'
  const isSeries = d.media_type === 'tv'
  const overview = d.overview || title.overview

  if (!expanded) {
    return (
      <div className="tcard" ref={ref} role="button" tabIndex={0}
        onClick={onToggle}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onToggle()}>
        <div className="poster">
          {title.poster_url ? (
            <img src={title.poster_url} alt={title.title} loading="lazy" />
          ) : (
            <div className="ph">🎬</div>
          )}
          <span className="type-tag">{typeLabel(d.media_type, d.seasons)}</span>
          <Rating score={score} source={scoreSrc} />
        </div>
        <div className="body">
          <div className="name" title={title.title}>{title.title}</div>
          <div className="meta">
            {title.year && <span>{title.year}</span>}
            {(title.genres || []).slice(0, 2).map((g) => <span key={g}>{g}</span>)}
          </div>
          <div className="desc">{overview}</div>
          <div className="hint">View details ›</div>
        </div>
      </div>
    )
  }

  return (
    <div className="tcard open" ref={ref}>
      <button className="close-x" onClick={onToggle} title="Close">✕</button>

      {loading && (
        <div className="load-detail">
          <span className="mini-spin" /> Fetching full details…
        </div>
      )}

      {!loading && !d ? (
        <div className="load-detail">⚠️ Couldn’t load details for this title.</div>
      ) : (
        <div className="acc">
          <div className="acc-hero">
            {d.poster_url && <img src={d.poster_url} alt="" loading="lazy" />}
            <div className="h-main">
              <h3>{d.title}</h3>
              <div className="sub">
                {[d.year, typeLabel(d.media_type, d.seasons), d.language]
                  .filter(Boolean)
                  .join(' · ')}
                {d.tagline ? ` — “${d.tagline}”` : ''}
              </div>
              <div className="facts">
                <span className="fact">★ {score || '—'}/10 {scoreSrc}</span>
                {(d.genres || []).slice(0, 4).map((g) => (
                  <span className="fact" key={g}>{g}</span>
                ))}
                {mins(d.runtime) && <span className="fact">{mins(d.runtime)}</span>}
                {isSeries && d.episodes && (
                  <span className="fact">{d.episodes} episodes</span>
                )}
                {d.age_rating && <span className="fact">{d.age_rating}</span>}
                {(d.countries || []).slice(0, 2).map((c) => (
                  <span className="fact" key={c}>{c}</span>
                ))}
              </div>
            </div>
          </div>

          <Accordion title="Overview" defaultOpen>
            <p style={{ margin: 0 }}>{overview || 'No overview available.'}</p>
            {d.status && <p style={{ margin: '8px 0 0' }}>Status: {d.status}</p>}
            {d.networks?.length > 0 && (
              <p style={{ margin: '6px 0 0' }}>Network: {d.networks.join(', ')}</p>
            )}
          </Accordion>

          <Accordion title="Details">
            <div className="specs">
              {d.creators?.length > 0 && (
                <div><b>{isSeries ? 'Created by' : 'Countries'}</b>
                  <span>{d.creators.join(', ')}</span></div>
              )}
              <div><b>Released</b><span>{d.release_date || d.year || '—'}</span></div>
              <div><b>Language</b><span>{d.language || '—'}</span></div>
              {mins(d.runtime) && (
                <div><b>{isSeries ? 'Episode length' : 'Runtime'}</b><span>{mins(d.runtime)}</span></div>
              )}
              {isSeries && d.seasons && (
                <div><b>Seasons</b><span>{d.seasons}</span></div>
              )}
              {d.vote_count > 0 && (
                <div><b>Votes</b><span>{d.vote_count.toLocaleString()}</span></div>
              )}
              {d.studios?.length > 0 && (
                <div><b>Studio</b><span>{d.studios.join(', ')}</span></div>
              )}
              {d.imdb_id && (
                <div><b>IMDb</b>
                  <span>
                    <a href={`https://www.imdb.com/title/${d.imdb_id}/`}
                       target="_blank" rel="noopener noreferrer" style={{ color: '#9dc0ff' }}>
                      {d.imdb_id} ↗
                    </a>
                  </span>
                </div>
              )}
              {d.homepage && (
                <div><b>Website</b>
                  <span>
                    <a href={d.homepage} target="_blank" rel="noopener noreferrer"
                       style={{ color: '#9dc0ff' }}>Official site ↗</a>
                  </span>
                </div>
              )}
            </div>
            {d.keywords?.length > 0 && (
              <div className="crew-list" style={{ marginTop: 11 }}>
                {d.keywords.map((k) => <span key={k}>{k}</span>)}
              </div>
            )}
          </Accordion>

          {(d.cast?.length > 0 || d.crew?.length > 0) && (
            <Accordion title="Cast &amp; Crew">
              {d.cast?.length > 0 && (
                <div className="cast-grid">
                  {d.cast.slice(0, 10).map((c, i) => (
                    <div className="cast-card" key={c.name + i}>
                      {c.profile_url
                        ? <img src={c.profile_url} alt={c.name} loading="lazy" />
                        : <div className="cast-card"><div className="ph" style={{ width: 58, height: 58 }}>👤</div></div>}
                      <div className="n">{c.name}</div>
                      {c.character && <div className="c">{c.character}</div>}
                    </div>
                  ))}
                </div>
              )}
              {d.crew?.length > 0 && (
                <div className="crew-list" style={{ marginTop: 12 }}>
                  {d.crew.map((c, i) => (
                    <span key={c.name + i}>{c.job}: {c.name}</span>
                  ))}
                </div>
              )}
            </Accordion>
          )}

          {d.trailers?.length > 0 && (
            <Accordion title="Trailers &amp; Clips">
              <div className="trailers">
                {d.trailers.map((t) => (
                  <a key={t.key} className="trailer" href={t.url}
                     target="_blank" rel="noopener noreferrer">
                    <img src={t.thumb} alt="" loading="lazy" />
                    <span>{t.name}</span>
                  </a>
                ))}
              </div>
            </Accordion>
          )}

          <Accordion title="Where to Watch">
            <WatchLinks watch={d.watch || []} />
          </Accordion>

          {d.similar?.length > 0 && (
            <Accordion title="More Like This">
              <div className="sim-grid">
                {d.similar.slice(0, 8).map((s) => (
                  <div className="sim" key={s.key} onClick={() => onOpenSimilar(s.key)}>
                    {s.poster_url ? <img src={s.poster_url} alt={s.title} loading="lazy" />
                                  : <div className="ph" style={{ aspectRatio: '2/3', display: 'grid', placeItems: 'center', fontSize: 24 }}>🎬</div>}
                    <div className="t" title={s.title}>{s.title}</div>
                  </div>
                ))}
              </div>
            </Accordion>
          )}
        </div>
      )}
    </div>
  )
}
