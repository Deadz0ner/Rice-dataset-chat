import { useState } from "react";

import type { ChatMessage, ChatSource } from "../types/chat";

interface MessageBubbleProps {
  message: ChatMessage;
}

function parsePreview(preview: string): { label: string; value: string }[] {
  return preview
    .split(" | ")
    .map((segment) => {
      const idx = segment.indexOf(": ");
      if (idx === -1) return null;
      return { label: segment.slice(0, idx).trim(), value: segment.slice(idx + 2).trim() };
    })
    .filter((entry): entry is { label: string; value: string } => entry !== null && entry.value !== "");
}

const KEY_FIELDS = new Set([
  "Product Description",
  "Exporter",
  "Buyer",
  "Foreign Country",
  "Indian Port",
  "Quantity",
  "Unit",
  "Value FC",
  "Currency",
  "Date",
]);

function SourceCard({ source, rank }: { source: ChatSource; rank: number }) {
  const fields = parsePreview(source.preview);
  const keyFields = fields.filter((f) => KEY_FIELDS.has(f.label));
  const score = source.score;
  const pct = score != null ? Math.round(score * 100) : null;

  return (
    <div className="source-card">
      <div className="source-card-header">
        <span className="source-rank">#{rank}</span>
        <span className="source-id">{source.row_id}</span>
        {pct != null ? (
          <span className="source-score-badge">{pct}% match</span>
        ) : null}
      </div>
      {pct != null ? (
        <div className="source-bar-track">
          <div className="source-bar-fill" style={{ width: `${pct}%` }} />
        </div>
      ) : null}
      <dl className="source-fields">
        {keyFields.map((f) => (
          <div key={f.label} className="source-field">
            <dt>{f.label}</dt>
            <dd>{f.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const hasSources = message.sources && message.sources.length > 0;

  return (
    <article className={`message ${isUser ? "message-user" : "message-assistant"}`}>
      <div className="message-header">
        <span>{isUser ? "You" : "Rice Analyst"}</span>
      </div>
      <div className="message-body">
        <p>{message.content}</p>
        {message.note ? <p className="message-note">{message.note}</p> : null}
      </div>
      {hasSources ? (
        <div className="message-sources">
          <button
            className="sources-toggle"
            onClick={() => setSourcesOpen((prev) => !prev)}
          >
            <span>{sourcesOpen ? "Hide" : "Show"} sources ({message.sources!.length})</span>
            <span className={`sources-chevron ${sourcesOpen ? "sources-chevron-open" : ""}`}>
              &#9662;
            </span>
          </button>
          {sourcesOpen ? (
            <div className="sources-list">
              {message.sources!.map((source, i) => (
                <SourceCard key={source.row_id} source={source} rank={i + 1} />
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </article>
  );
}
