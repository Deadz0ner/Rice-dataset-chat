import { useEffect, useState } from "react";

import { ChatInput } from "../components/ChatInput";
import { ErrorModal } from "../components/ErrorModal";
import { LoadingDots } from "../components/LoadingDots";
import { MessageBubble } from "../components/MessageBubble";
import { UploadPanel } from "../components/UploadPanel";
import { useAutoScroll } from "../hooks/useAutoScroll";
import { sendChatMessage } from "../services/chatService";
import { fetchDatasetSummary } from "../services/datasetService";
import type { ChatMessage } from "../types/chat";
import type { DatasetSummary } from "../types/dataset";

const starterMessage: ChatMessage = {
  id: "assistant-welcome",
  role: "assistant",
  content:
    "Ask questions in natural language about the rice EXIM dataset. All answers are grounded strictly in the loaded data.",
};

export function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([starterMessage]);
  const [datasetSummary, setDatasetSummary] = useState<DatasetSummary | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useAutoScroll(messages, isLoading);

  useEffect(() => {
    async function loadSummary() {
      try {
        const summary = await fetchDatasetSummary();
        setDatasetSummary(summary);
      } catch {
        setDatasetSummary(null);
      }
    }

    void loadSummary();
  }, []);

  async function handleSend(message: string) {
    setError(null);
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: message,
    };

    setMessages((current) => [...current, userMessage]);
    setIsLoading(true);

    try {
      const history = messages
        .filter((m) => m.id !== "assistant-welcome")
        .map((m) => ({ role: m.role, content: m.content }));
      const response = await sendChatMessage({ message, history });
      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: response.answer,
        note: response.note,
        sources: response.sources,
      };
      setMessages((current) => [...current, assistantMessage]);
    } catch (chatError) {
      setError(chatError instanceof Error ? chatError.message : "Unable to fetch response.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <UploadPanel datasetSummary={datasetSummary} />

      <section className="chat-panel">
        <header className="chat-header">
          <div>
            <p className="eyebrow">Grounded chat</p>
            <h1>Rice Dataset Assistant</h1>
          </div>
          <p className="chat-subtitle">
            Ask questions in plain English about the rice EXIM dataset.
          </p>
        </header>

        <div className="chat-window">
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
          {isLoading ? (
            <div className="message message-assistant">
              <div className="message-header">
                <span>Rice Analyst</span>
              </div>
              <LoadingDots />
            </div>
          ) : null}
          <div ref={scrollRef} />
        </div>

        <ChatInput disabled={isLoading} onSubmit={handleSend} />
      </section>

      {error ? <ErrorModal message={error} onClose={() => setError(null)} /> : null}
    </main>
  );
}
