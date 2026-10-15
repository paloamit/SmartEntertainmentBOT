export default function Sidebar({
  sessions, activeId, limits, onSelect, onNew, onDelete, onSync, syncing,
}) {
  const atCap = sessions.length >= (limits?.max_sessions ?? 5)
  return (
    <aside className="sidebar">
      <button className="new-chat" onClick={onNew} title={atCap ? 'Delete a chat to start a new one' : 'Start a new chat'}>
        + New chat
      </button>

      <div className="sidebar-title">Your chats</div>

      <div className="session-list">
        {sessions.length === 0 && <div className="muted" style={{ fontSize: 13, padding: 8 }}>No chats yet</div>}
        {sessions.map((s) => (
          <div
            key={s.id}
            className={`session ${s.id === activeId ? 'active' : ''}`}
            onClick={() => onSelect(s.id)}
            title={s.title}
          >
            <span className="session-name">{s.title}</span>
            {s.query_count > 0 && (
              <span className="session-q">{s.query_count}/{limits?.max_queries_per_session ?? 10}</span>
            )}
            <button
              className="del"
              title="Delete chat"
              onClick={(e) => { e.stopPropagation(); onDelete(s.id) }}
            >
              ✕
            </button>
          </div>
        ))}
      </div>

      <button className="sync-btn" onClick={onSync}>
        {syncing ? 'Sync code active' : '🔗 Sync across devices'}
      </button>

      <div className="sidebar-foot">
        {sessions.length} of {limits?.max_sessions ?? 5} chats ·{' '}
        {limits?.max_queries_per_session ?? 10} questions each
      </div>
    </aside>
  )
}
