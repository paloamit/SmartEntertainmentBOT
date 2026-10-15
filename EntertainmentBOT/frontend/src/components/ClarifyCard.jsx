import { useState } from 'react'

/**
 * The human-in-the-loop question. The bot decides which filters it still needs
 * and how to ask for each one:
 *   dropdown → many options, pick one
 *   radio    → 2-4 options, pick one
 *   checkbox → several may apply
 * Answers are submitted back as structured values, not parsed from prose.
 */
export default function ClarifyCard({ clarify, answered, onSubmit }) {
  const [values, setValues] = useState(() => {
    const init = {}
    ;(clarify.fields || []).forEach((f) => {
      init[f.name] = f.ui === 'checkbox' ? [] : ''
    })
    return init
  })
  const [busy, setBusy] = useState(false)

  const fields = clarify?.fields || []
  const quick = clarify?.quick_options || []
  const chosen = Object.values(values).filter((v) =>
    Array.isArray(v) ? v.length : String(v || '').trim(),
  ).length

  const setOne = (name, value) => setValues((v) => ({ ...v, [name]: value }))
  const toggle = (name, value) =>
    setValues((v) => {
      const list = v[name] || []
      return {
        ...v,
        [name]: list.includes(value) ? list.filter((x) => x !== value) : [...list, value],
      }
    })

  const submit = (extra = {}) => {
    if (busy) return
    setBusy(true)
    const payload = {}
    Object.entries(values).forEach(([k, v]) => {
      if (Array.isArray(v) ? v.length : String(v || '').trim()) payload[k] = v
    })
    onSubmit({ answers: { ...payload, ...extra } })
  }

  if (answered) {
    return (
      <div className="cf-answered">
        <span>🎬</span>
        <span>
          Based on your inputs, here are my recommended picks — best match first.
        </span>
      </div>
    )
  }

  return (
    <div className="clarify">
      <div className="cf-grid">
        {fields.map((f) => (
          <div
            className={`cf-field ${f.ui !== 'dropdown' && f.options.length >= 4 ? 'cf-wide' : ''}`}
            key={f.name}
          >
            <label className="cf-label">{f.label}</label>

            {f.ui === 'dropdown' ? (
              <select
                className="cf-drop"
                value={values[f.name] || ''}
                onChange={(e) => setOne(f.name, e.target.value)}
              >
                <option value="">Any</option>
                {f.options.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            ) : (
              <div className="cf-opts">
                {f.options.map((o) => {
                  const on = f.ui === 'checkbox'
                    ? (values[f.name] || []).includes(o.value)
                    : values[f.name] === o.value
                  return (
                    <button
                      key={o.value}
                      className={`opt ${on ? 'on' : ''}`}
                      onClick={() =>
                        f.ui === 'checkbox' ? toggle(f.name, o.value) : setOne(f.name, o.value)
                      }
                    >
                      {o.label}
                    </button>
                  )
                })}
              </div>
            )}
          </div>
        ))}
      </div>

      {quick.length > 0 && (
        <div className="cf-actions cf-quick" style={{ flexWrap: 'wrap' }}>
          {quick.map((q) => (
            <button key={q} className="chip" disabled={busy}
              onClick={() => submit({ _quick: q })}>
              {q}
            </button>
          ))}
        </div>
      )}

      <div className="cf-actions">
        {/* Leaving everything unselected is a valid answer — "no preference". */}
        <button className="cf-submit" disabled={busy} onClick={() => submit()}>
          {busy ? 'Finding…' : chosen > 0 ? 'Show me 🎬' : 'Just recommend 🎬'}
        </button>
        <button className="cf-skip" disabled={busy} onClick={() => submit()}>
          Surprise me
        </button>
      </div>
    </div>
  )
}
