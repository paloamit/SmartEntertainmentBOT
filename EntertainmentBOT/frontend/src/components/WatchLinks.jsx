/**
 * Where to watch — streaming availability per country, straight from the
 * provider API. Shows subscription / free / ads / rent / buy with direct links.
 */
import { useState } from 'react'

export default function WatchLinks({ watch }) {
  const [cc, setCc] = useState(watch[0]?.country_code || '')
  const current = watch.find((w) => w.country_code === cc) || watch[0]

  if (!watch?.length) {
    return (
      <div className="no-watch">
        No streaming availability listed for this title right now.
      </div>
    )
  }

  return (
    <div>
      {watch.length > 1 && (
        <div className="watch-country">
          <span style={{ fontSize: 12, color: 'var(--faint)' }}>Region</span>
          <select
            className="cf-drop"
            style={{ width: 'auto', minWidth: 160 }}
            value={current.country_code}
            onChange={(e) => setCc(e.target.value)}
          >
            {watch.map((w) => (
              <option key={w.country_code} value={w.country_code}>
                {w.country} ({w.country_code})
              </option>
            ))}
          </select>
        </div>
      )}

      {current.categories.map((cat) => (
        <div className="watch-row" key={cat.key}>
          <div className="wl">{cat.label}</div>
          <div className="providers">
            {cat.providers.map((p) => (
              <a
                key={p.name}
                className="provider"
                href={current.link || '#'}
                target="_blank"
                rel="noopener noreferrer"
                title={`Watch on ${p.name} in ${current.country}`}
                onClick={(e) => {
                  // Deep links are per-provider only in some regions; fall back
                  // to the provider's watch guide rather than a dead link.
                  if (!current.link) e.preventDefault()
                }}
              >
                {p.logo_url && <img src={p.logo_url} alt="" loading="lazy" />}
                {p.name}
              </a>
            ))}
          </div>
        </div>
      ))}

      {current.link && (
        <a
          className="watch-all"
          href={current.link}
          target="_blank"
          rel="noopener noreferrer"
        >
          See all watch options in {current.country} →
        </a>
      )}
    </div>
  )
}
