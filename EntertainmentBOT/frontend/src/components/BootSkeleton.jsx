/** First-paint skeleton, shown while /api/bootstrap (greeting + sessions) loads. */
export default function BootSkeleton() {
  return (
    <div className="messages">
      <div className="msg assistant greeting">
        <div className="bubble skel-bubble">
          <span className="skel-line w-70" />
          <span className="skel-line w-90" />
          <span className="skel-line w-45" />
        </div>
      </div>
      <div className="msg assistant">
        <div className="bubble skel-bubble">
          <span className="skel-line w-85" />
          <span className="skel-line w-60" />
        </div>
      </div>
      <div className="titles">
        {[0, 1, 2, 3, 4].map((i) => (
          <div key={i} className="tcard">
            <div className="poster skel-poster" />
            <div className="body">
              <span className="skel-line w-80" />
              <span className="skel-line w-45" />
            </div>
          </div>
        ))}
      </div>
      <div className="msg user">
        <div className="bubble">
          <span className="skel-line w-50" />
        </div>
      </div>
      <div className="msg assistant">
        <div className="bubble skel-bubble">
          <span className="skel-line w-75" />
        </div>
      </div>
    </div>
  )
}
