import type { ChatMessage } from "../types/chat";

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <article className={`message ${isUser ? "message-user" : "message-assistant"}`}>
      <div className="message-header">
        <span>{isUser ? "You" : "Rice Analyst"}</span>
      </div>
      <p>{message.content}</p>
      {message.note ? <p className="message-note">{message.note}</p> : null}
      {message.sources && message.sources.length > 0 ? (
        <div className="message-sources">
          <strong>Sources</strong>
          {message.sources.map((source) => (
            <div key={source.row_id} className="source-card">
              <div className="source-row">
                <span>{source.row_id}</span>
                {typeof source.score === "number" ? <span>{source.score.toFixed(2)}</span> : null}
              </div>
              <p>{source.preview}</p>
            </div>
          ))}
        </div>
      ) : null}
    </article>
  );
}
