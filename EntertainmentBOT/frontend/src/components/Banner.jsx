export default function Banner({ tone = 'info', icon = 'ℹ️', text, action, onAction, onClose }) {
  if (!text) return null
  return (
    <div className={`banner ${tone}`}>
      <span className="b-ico">{icon}</span>
      <div className="b-body">
        <div>{text}</div>
        {action && (
          <button className="link" onClick={onAction}>{action}</button>
        )}
      </div>
      <button className="b-close" onClick={onClose} title="Dismiss">✕</button>
    </div>
  )
}
